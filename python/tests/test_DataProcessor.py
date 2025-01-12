import unittest
from cdt.coditT5.DataProcessor import DataProcessor, RawData

class TestDataProcessor(unittest.TestCase):

    def test_process_raw_data_point(self):
        dp = DataProcessor()

        # Test case 1: Normal case
        raw_data = RawData(
            source="class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, World!\")",
            target="class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, World!\") ; } } ;",
            context="Complete this code to print Hello, World!"
        )
        input_str, edit_str, target_str = dp.process_raw_data_point(raw_data)
        self.assertEqual(input_str, "class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, World!\") </s> Complete this code to print Hello, World!")
        self.assertEqual(edit_str, "<INSERT> ; } } ; <INSERT_END> <s> class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, World!\") ; } } ;")
        self.assertEqual(target_str, "class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, World!\") ; } } ;")

    def test_remove_edits(self):

        # Test case 1: Normal case
        input_str = "<INSERT> ; } } ; <INSERT_END> <s> class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, World!\") ; } } ;"
        output_str = DataProcessor.remove_edits(input_str)
        self.assertEqual(output_str, "<s> class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, World!\") ; } } ;")

if __name__ == '__main__':
    unittest.main()