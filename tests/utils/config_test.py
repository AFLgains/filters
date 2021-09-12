# -*- coding: utf-8 -*-
import unittest
import unittest.mock as mock

from filters.utils.config import Config


class TestConfig(unittest.TestCase):
    def test_load_invalid_file(self):
        config = Config("fake.yml")
        self.assertRaises(FileNotFoundError, config.read)

    @mock.patch("os.path.isfile")
    @mock.patch("os.access")
    def test_load_file(self, mock_isfile, mock_access):
        mock_isfile.return_value = True
        mock_access.return_value = True

        data = """
        logging:
          version: 1
        """
        with mock.patch("builtins.open", mock.mock_open(read_data=data)):
            config = Config("").read()
            self.assertEqual(1, config.logging.version)
            self.assertIsNone(config.get("fake"))
