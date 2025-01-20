#!/usr/bin/env python
"""
Main module for pdf processing
"""
import asyncio
import hashlib
import json
import logging
import multiprocessing
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Dict

import aiofiles
import fitz
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Initialize logging
LOG_DIR = Path("src/log")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, f"log_{date.today()}.log")
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class VisionPDFProcessor:
    """Main class to process pdfs"""

    def __init__(
        self,
        input_dir: str = "src/files_to_process",
        output_dir: str = "src/data/json",
        images_dir: str = "src/data/processed_images",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = Path(images_dir)
        self.max_workers = multiprocessing.cpu_count()
        self.io_executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Ensure directories exist
        self._setup_directories()

    def _setup_directories(self):
        for dir_path in [self.input_dir, self.output_dir, self.images_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def append_to_json_file(
        self, file_path, file_hash, pdf_path, text, num_pages
    ):
        """The JSON file contains the two interpretations of the pdf"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {
                "file_path": str(pdf_path),
                "file_hash": file_hash,
                "processed_date": datetime.now().isoformat(),
                "total_pages": num_pages,
                "processed": "True",
                "name": str(os.path.basename(pdf_path)),
                "content": text,
            }
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    async def process_single_pdf(self, pdf_path: Path) -> Dict:
        """Process each pdf"""
        file_hash = await self._calculate_file_hash(pdf_path)
        output_path = (
            self.output_dir / f"{pdf_path.stem}_{file_hash[:8]}.json"
        )

        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("file_hash") == file_hash:
                    # logger.info(f"Already processed {pdf_path}")
                    return (pdf_path, True, 2)

        images_path = await self._extract_images(pdf_path, file_hash)
        if not images_path:
            logger.error("No images path")
            return (pdf_path, False, 3)

        num_images = len(list(images_path.glob("*.*")))
        ocr_text = await self._vision_process_page(images_path)
        pdf_text = await self.extract_text_from_pdf(pdf_path)

        ocr_text = " ".join(ocr_text)
        ocr_text = await self.filter_special_chars(ocr_text)
        text = {"ocr": ocr_text}
        text["pdf_text"] = pdf_text

        self.append_to_json_file(
            output_path, file_hash, pdf_path, text, num_images
        )

        self._cleanup_images(file_hash)

        return (pdf_path, True, 1)

    async def filter_special_chars(self, text):
        """Clean up the gather text from too many spaces
        and useless special chars"""
        cleaned_text = re.sub(r"[\t\n\r]", " ", text)

        # Step 2: Replace multiple spaces with a single space
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        return cleaned_text

    async def extract_text_from_pdf(self, pdf_path):
        """Uses OCR and pdf extraction to get the text from the pdf"""
        text = ""
        try:
            with fitz.open(pdf_path) as pdf_doc:
                for page in pdf_doc:
                    text += page.get_text()

            if text:

                return await self.filter_special_chars(text)
            else:
                return " "

        except Exception as e:
            logger.error(" Extracting pdf failed %s", e)
            return " "

    def _cleanup_images(self, pdf_hash: str):
        image_dir = self.images_dir / pdf_hash
        if image_dir.exists():
            try:
                shutil.rmtree(image_dir)
                # logger.info(f"Removed processed images for hash {pdf_hash}")
            except Exception as e:
                logger.error(
                    "Failed to remove processed images for hash %s - %s  ",
                    pdf_hash,
                    e,
                )
        else:
            logger.warning(
                "No images directory found for hash %s", pdf_hash
            )

    async def _extract_images(self, pdf_path: Path, pdf_hash: str):
        image_dir = self.images_dir / pdf_hash
        if image_dir.exists():

            return image_dir

        image_dir.mkdir(parents=True, exist_ok=True)

        images = await asyncio.to_thread(
            convert_from_path,
            pdf_path,
            dpi=100,
            fmt="png",
            thread_count=self.max_workers,
            use_pdftocairo=True,
        )

        async def save_image(image, index):
            image_path = image_dir / f"{pdf_path.stem}_{index:03d}.jpg"
            image.save(image_path, "jpeg")

        await asyncio.gather(
            *[save_image(img, idx + 1) for idx, img in enumerate(images)]
        )

        return image_dir

    async def _vision_process_page(self, images_path):
        async def process_image(image_path):
            with Image.open(image_path) as image:
                return pytesseract.image_to_string(image)

        tasks = []
        for img_path in sorted(
            images_path.glob("*"),
            key=lambda x: int(re.findall(r"\d+", x.stem)[0]),
        ):
            if isinstance(img_path, Path):  # Check if it's a Path object
                if img_path.suffix.lower() in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".bmp",
                ]:
                    tasks.append(process_image(img_path))
            else:
                logger.error(
                    "Unexpected type for image path: %s", type(img_path)
                )
                continue

        return await asyncio.gather(*tasks)

    @staticmethod
    async def _calculate_file_hash(filepath: Path) -> str:
        sha256_hash = hashlib.sha256()
        async with aiofiles.open(filepath, "rb") as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
