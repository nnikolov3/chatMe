#!/usr/bin/env python
import os
import base64
import asyncio
import logging
import hashlib
import json
from pathlib import Path
from datetime import datetime
import re
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
import aiofiles
from PIL import Image
from pdf2image import convert_from_path
from ollama import AsyncClient
import pytesseract
import easyocr
import multiprocessing

# Initialize logging
log_dir = "src/log"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
reader = easyocr.Reader(["en"])


class VisionPDFProcessor:

    def __init__(
        self,
        input_dir: str = "src/files_to_process",
        output_dir: str = "src/data/json",
        images_dir: str = "src/data/processed_images",
        max_workers: int = multiprocessing.cpu_count(),
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = Path(images_dir)
        self.max_workers = multiprocessing.cpu_count()
        self.io_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.vision_model = "minicpm-v"

        # Ensure directories exist
        self._setup_directories()

        logger.info(
            f"Initialized VisionPDFProcessor with input_dir={self.input_dir}, "
            f"output_dir={self.output_dir}, max_workers={self.max_workers}"
        )

    def _setup_directories(self):
        for dir_path in [self.input_dir, self.output_dir, self.images_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def append_to_json_file(self, file_path, file_hash, pdf_path, text, num_pages):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {
                "file_path": str(pdf_path),
                "file_hash": file_hash,
                "processed_date": datetime.now().isoformat(),
                "content": {"pages": text, "total_pages": num_pages},
            }
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    async def process_single_pdf(self, pdf_path: Path) -> Dict:
        file_hash = await self._calculate_file_hash(pdf_path)
        output_path = self.output_dir / f"{pdf_path.stem}_{file_hash[:8]}.json"

        if output_path.exists():
            with open(output_path, "r") as f:
                data = json.load(f)
                if data.get("file_hash") == file_hash:
                    logger.info(f"Already processed {pdf_path}")
                    return (pdf_path, True, 2)

        images_path = await self._extract_images(pdf_path, file_hash)
        if not images_path:
            logger.error("No images path")
            return (pdf_path, False, 3)

        num_images = len(list(images_path.glob("*.*")))
        text = await self._vision_process_page(images_path, num_images)

        self.append_to_json_file(output_path, file_hash, pdf_path, text, num_images)
        return (pdf_path, True, 1)

    async def _extract_images(self, pdf_path: Path, pdf_hash: str):
        image_dir = self.images_dir / pdf_hash
        if image_dir.exists():
            logger.info(f"Images already processed: {image_dir}")
            return image_dir

        image_dir.mkdir(parents=True, exist_ok=True)

        images = await asyncio.to_thread(
            convert_from_path,
            pdf_path,
            dpi=300,
            fmt="jpeg",
            thread_count=self.max_workers,
            transparent=True,
            use_pdftocairo=True,
        )

        async def save_image(image, index):
            image_path = image_dir / f"{pdf_path.stem}_{index:03d}.jpg"
            image.save(image_path, "jpeg")

        await asyncio.gather(
            *[save_image(img, idx + 1) for idx, img in enumerate(images)]
        )

        return image_dir

    async def _vision_process_page(self, images_path, num_images):
        async def process_image(image_path, i):
            with Image.open(image_path) as image:
                text = pytesseract.image_to_string(image)
                page_data = {"ocr_text": text}

                with open(image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                content = """Provide an exhaustive and meticulous transcription of this document, emphasizing accuracy and detail. Capture every detail comprehensively to its extent, ensuring precision in your answer. Highlight graphs, charts, and code blocks, transcribing their content accurately and in full."""
                my_message = [
                    {"role": "user", "content": content, "images": [encoded_image]}
                ]

                response = await AsyncClient().chat(
                    model="minicpm-v", messages=my_message
                )
                page_data["first_vision_analysis"] = response["message"]["content"]

                response_2 = reader.readtext(image, detail=0)

                page_data["second_vision_analysis"] = response_2

            return {f"page_{i}": page_data}

        tasks = [
            process_image(img_path, i)
            for i, img_path in enumerate(
                sorted(
                    images_path.glob("*"),
                    key=lambda x: int(re.findall(r"\d+", x.stem)[0]),
                )
            )
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]
        ]
        return await asyncio.gather(*tasks)

    @staticmethod
    async def _calculate_file_hash(filepath: Path) -> str:
        sha256_hash = hashlib.sha256()
        async with aiofiles.open(filepath, "rb") as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()


if __name__ == "__main__":
    processor = VisionPDFProcessor()
    asyncio.run(processor.process_pdfs())
