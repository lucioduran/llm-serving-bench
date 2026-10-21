import json
import unittest
from pathlib import Path

from serving_bench.experiment import verify

ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_readme_contract(self):
        headings = [
            line[3:]
            for line in (ROOT / "README.md").read_text().splitlines()
            if line.startswith("## ")
        ]
        self.assertEqual(
            headings,
            [
                "Problem",
                "Scope",
                "Quick start",
                "Usage",
                "Architecture",
                "Reproduction",
                "Results",
                "Testing",
                "Limitations",
                "Contributing",
                "License",
            ],
        )

    def test_published_evidence_and_comparable_token_inputs(self):
        path = ROOT / "results/reference/report.json"
        if not path.exists():
            self.skipTest("Reference experiment has not been recorded")
        report = json.loads(path.read_text())
        verify(report, ROOT / "data/prompts.json")
        self.assertFalse(report["source_dirty"])
        self.assertEqual(report["model"]["dtype"], "float32")
        inputs = {}
        for cell in report["cells"]:
            self.assertEqual(cell["summary"]["errors"], 0)
            for index, row in enumerate(cell["requests"]):
                key = (cell["context_tokens"], index)
                ids = row["completion"]["input_ids"]
                self.assertEqual(inputs.setdefault(key, ids), ids)

    def test_no_em_dash_in_project_text(self):
        for folder in ["serving_bench", "tests", "docs", "data"]:
            for path in (ROOT / folder).rglob("*"):
                if path.suffix in {".py", ".md", ".json"}:
                    self.assertNotIn(chr(0x2014), path.read_text(), str(path))
