import unittest

from semantic_books.generation_service import OllamaGenerator


class GenerationServiceTests(unittest.TestCase):
    def test_strip_thinking_sections_removes_think_tags(self) -> None:
        raw = "<think>internal reasoning</think>\nAnswer: SQL is a query language. [C1]\nSourcesUsed: C1"
        cleaned = OllamaGenerator._strip_thinking_sections(raw)
        self.assertNotIn("<think>", cleaned)
        self.assertIn("Answer: SQL is a query language. [C1]", cleaned)

    def test_strip_thinking_sections_drops_repeated_reasoning_loop(self) -> None:
        loop = (
            "Generating grounded answer...\n\n"
            "So, I need to create a 2-week learning path using only the provided books.\n\n"
            "But the user wants a 2-week path, so maybe 14 days.\n\n"
            "So, I need to create a 2-week learning path using only the provided books.\n\n"
            "But the user wants a 2-week path, so maybe 14 days.\n\n"
            "Answer: Week 1 focuses on operating systems [C1].\n"
            "Week 2 focuses on AI and ML workflow [C3][C5].\n"
            "SourcesUsed: C1, C3, C5"
        )
        cleaned = OllamaGenerator._strip_thinking_sections(loop)
        self.assertNotIn("So, I need to create a 2-week learning path", cleaned)
        self.assertNotIn("But the user wants a 2-week path", cleaned)
        self.assertIn("Answer: Week 1 focuses on operating systems [C1].", cleaned)
        self.assertIn("SourcesUsed: C1, C3, C5", cleaned)


if __name__ == "__main__":
    unittest.main()
