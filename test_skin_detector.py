import unittest
from skin_detector import ExplicitSkinDetector


class ExplicitSkinDetectorTests(unittest.TestCase):

    def setUp(self):
        self.explicit_skin_detector = ExplicitSkinDetector()

    def test_is_skin_pixel(self):
        is_pixel = self.explicit_skin_detector.is_skin_pixel((148, 205, 255))
        self.assertEqual(is_pixel, True)

    def test_is_skin_pixel_is_not_a_pixel(self):
        is_pixel = self.explicit_skin_detector.is_skin_pixel((1, 20, 30))
        self.assertEqual(is_pixel, False)
