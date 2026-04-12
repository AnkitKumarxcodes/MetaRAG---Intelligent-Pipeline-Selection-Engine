# loader.py

import os
import json
import requests
from typing import List, Optional, Union


class DocumentLoader:
    def __init__(
        self,
        data_path: Optional[Union[str, List[str]]] = None,
        recursive: bool = True,             # traverse subdirectories
    ):
        # Accept a single path or a list of paths
        if data_path is None:
            self.data_paths = []
        elif isinstance(data_path, str):
            self.data_paths = [data_path]
        else:
            self.data_paths = data_path

        self.recursive = recursive

    # ─────────────────────────────────────────
    # CORE UTILITY — file discovery
    # ─────────────────────────────────────────

    def _iter_files(self, extensions: tuple) -> List[str]:
        """
        Yield all file paths matching given extensions across
        all data_paths, respecting the recursive flag.
        """
        matched = []
        for root_path in self.data_paths:
            root_path = os.path.abspath(root_path)

            # Single file passed directly
            if os.path.isfile(root_path):
                if root_path.endswith(extensions):
                    matched.append(root_path)
                continue

            if not os.path.isdir(root_path):
                print(f"[Warning] Path not found: {root_path}")
                continue

            if self.recursive:
                for dirpath, _, filenames in os.walk(root_path):
                    for f in filenames:
                        if f.endswith(extensions):
                            matched.append(os.path.join(dirpath, f))
            else:
                for f in os.listdir(root_path):
                    full = os.path.join(root_path, f)
                    if os.path.isfile(full) and f.endswith(extensions):
                        matched.append(full)

        return matched

    # ─────────────────────────────────────────
    # FILE LOADERS
    # ─────────────────────────────────────────

    def load_txt(self) -> List[str]:
        docs = []
        for path in self._iter_files((".txt",)):
            with open(path, "r", encoding="utf-8") as f:
                docs.append(f.read())
        return docs

    def load_pdf(self) -> List[str]:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("Run: pip install pypdf")

        docs = []
        for path in self._iter_files((".pdf",)):
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                docs.append(text)
            else:
                print(f"[Warning] Scanned PDF, no text extracted: {path}")
        return docs

    def load_html(self) -> List[str]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Run: pip install beautifulsoup4")

        docs = []
        for path in self._iter_files((".html", ".htm")):
            with open(path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                if text:
                    docs.append(text)
        return docs

    def load_docx(self) -> List[str]:
        try:
            from docx import Document
        except ImportError:
            raise ImportError("Run: pip install python-docx")

        docs = []
        for path in self._iter_files((".docx",)):
            doc = Document(path)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            if text:
                docs.append(text)
        return docs

    def load_csv(self, text_columns: Optional[List[str]] = None) -> List[str]:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Run: pip install pandas")

        docs = []
        for path in self._iter_files((".csv",)):
            df = pd.read_csv(path)
            cols = text_columns if text_columns else df.select_dtypes(include="object").columns.tolist()
            for _, row in df.iterrows():
                row_text = " | ".join(
                    f"{col}: {row[col]}" for col in cols
                    if col in row and pd.notna(row[col])
                )
                if row_text.strip():
                    docs.append(row_text)
        return docs

    def load_json(self, text_key: Optional[str] = None) -> List[str]:
        docs = []
        for path in self._iter_files((".json",)):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if text_key and isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and text_key in item:
                        docs.append(str(item[text_key]))
            else:
                docs.append(json.dumps(data, indent=2))
        return docs

    # ─────────────────────────────────────────
    # WEB LOADERS
    # ─────────────────────────────────────────

    def load_url(self, url: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Run: pip install beautifulsoup4")

        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        if "application/pdf" in response.headers.get("Content-Type", ""):
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(response.content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

    def load_urls(self, urls: List[str]) -> List[str]:
        docs = []
        for url in urls:
            try:
                text = self.load_url(url)
                if text.strip():
                    docs.append(text)
                    print(f"[✓] {url}")
            except Exception as e:
                print(f"[✗] Failed {url}: {e}")
        return docs

    def load_sitemap(self, sitemap_url: str, max_pages: int = 20) -> List[str]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Run: pip install beautifulsoup4")

        response = requests.get(sitemap_url, timeout=10)
        soup = BeautifulSoup(response.text, "xml")
        urls = [loc.text for loc in soup.find_all("loc")][:max_pages]
        print(f"[Sitemap] Found {len(urls)} URLs, loading up to {max_pages}...")
        return self.load_urls(urls)

    # ─────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────

    def summary(self):
        """Print a tree of all discovered files across all paths."""
        print("\n[DocumentLoader] File Discovery Summary")
        print("=" * 45)
        all_exts = (".txt", ".pdf", ".html", ".htm", ".docx", ".csv", ".json")
        for path in self.data_paths:
            print(f"\n📁 {os.path.abspath(path)}")
            files = self._iter_files(all_exts)
            by_dir = {}
            for f in files:
                folder = os.path.dirname(f)
                by_dir.setdefault(folder, []).append(os.path.basename(f))
            for folder, fnames in by_dir.items():
                rel = os.path.relpath(folder, path)
                print(f"  └─ {rel}/")
                for name in fnames:
                    print(f"       • {name}")
        print(f"\nTotal paths: {len(self.data_paths)} | Recursive: {self.recursive}")
        print("=" * 45)

    # ─────────────────────────────────────────
    # UNIFIED ENTRY POINT
    # ─────────────────────────────────────────

    def load_all(
        self,
        urls: Optional[List[str]] = None,
        sitemap_url: Optional[str] = None,
        json_text_key: Optional[str] = None,
        csv_text_columns: Optional[List[str]] = None,
    ) -> List[str]:
        docs = []

        if self.data_paths:
            loaders = [
                ("TXT",  self.load_txt),
                ("PDF",  self.load_pdf),
                ("HTML", self.load_html),
                ("DOCX", self.load_docx),
                ("CSV",  lambda: self.load_csv(csv_text_columns)),
                ("JSON", lambda: self.load_json(json_text_key)),
            ]
            for name, loader in loaders:
                try:
                    loaded = loader()
                    docs.extend(loaded)
                    if loaded:
                        print(f"[✓] {name}: {len(loaded)} doc(s)")
                except Exception as e:
                    print(f"[✗] {name} loader failed: {e}")

        if urls:
            docs.extend(self.load_urls(urls))

        if sitemap_url:
            docs.extend(self.load_sitemap(sitemap_url))

        print(f"\n[DocumentLoader] Total docs loaded: {len(docs)}")
        return docs