import unittest

from router import detect_document_type, get_document_label
from utils.report_gen import generate_pdf_report


class RouterTests(unittest.TestCase):
    def test_detect_document_type_for_aadhaar(self):
        text = "Unique Identification Authority of India Aadhaar No"
        self.assertEqual(detect_document_type(text), "aadhaar")

    def test_get_document_label_for_invoice(self):
        self.assertEqual(get_document_label("invoice"), "Invoice / Bill")


class ReportTests(unittest.TestCase):
    def test_generate_pdf_report_returns_bytes(self):
        pdf = generate_pdf_report([
            {
                "filename": "sample.txt",
                "doc_label": "General Document",
                "fields": {"Name": "Test User"},
                "raw_text": "Sample document text",
            }
        ])
        self.assertIsInstance(pdf, bytes)
        self.assertGreater(len(pdf), 0)


if __name__ == "__main__":
    unittest.main()