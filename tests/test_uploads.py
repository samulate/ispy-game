import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image


# Route tests should run without downloading or loading the AI models.
detector = ModuleType('object_detection_with_huggingface')
detector.filepath_or_image_to_all_data = Mock()
detector.text_to_embedding = Mock()
detector.compute_similarity = Mock()
with patch.dict(sys.modules, {'object_detection_with_huggingface': detector}):
    import app as web


class UploadTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='ispy-test-')
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        config = patch.dict(web.app.config, TESTING=True, UPLOAD_PATH=str(self.directory))
        config.start()
        self.addCleanup(config.stop)
        self.client = web.app.test_client()
        self.result = Mock()
        self.result.save.side_effect = lambda filename: Image.new('RGB', (10, 10)).save(filename)
        process = patch.object(web, 'filepath_or_image_to_all_data', return_value=(
            'red', np.array([255, 0, 0]), None, None, 'bus', self.result,
            np.zeros((10, 10, 3), dtype=np.uint8),
        ))
        self.process = process.start()
        self.addCleanup(process.stop)

    def upload(self, image_format='PNG', extension='png'):
        image = io.BytesIO()
        Image.new('RGB', (10, 10), 'red').save(image, format=image_format)
        image.seek(0)
        return self.client.post('/', data={'file-upload': (image, f'photo.{extension}')},
                                follow_redirects=True)

    def assert_round_images(self):
        with self.client.session_transaction() as session:
            files = session['files']
            self.assertEqual(len(files), 3)
            self.assertEqual(session['name'], 'bus')
            self.assertFalse(session['error'])
        for filename in files:
            self.assertTrue((self.directory / filename).is_file())
            response = self.client.get(f'/uploads/{filename}')
            self.assertEqual(response.status_code, 200)
            response.close()
        return files

    def test_png_and_jpeg_uploads_serve_all_images(self):
        for image_format, extension in [('PNG', 'png'), ('JPEG', 'jpg'), ('JPEG', 'jpeg')]:
            with self.subTest(extension=extension):
                response = self.upload(image_format, extension)
                self.assertEqual(response.status_code, 200)
                self.assertIn(b'Show Boxes', response.data)
                self.assert_round_images()
                self.assertEqual(len(list(self.directory.iterdir())), 3)

    def test_spy_again_reuses_original_and_resets_round(self):
        self.upload()
        original_files = self.assert_round_images()
        with self.client.session_transaction() as session:
            session.update(points=20, guesses=3, hint_used=True, similar=True)
        response = self.client.post('/again', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.assert_round_images(), original_files)
        with self.client.session_transaction() as session:
            self.assertEqual(session['points'], 20)
            self.assertEqual(session['guesses'], 0)
            self.assertFalse(session['hint_used'])
            self.assertIsNone(session['similar'])

    def test_no_detections_shows_error_without_missing_directory_crash(self):
        self.process.side_effect = ValueError('No objects detected')
        with self.assertLogs(web.app.logger, level='ERROR'):
            response = self.upload()
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"We couldn't detect any objects", response.data)
        self.assertEqual(list(self.directory.iterdir()), [])
        with self.client.session_transaction() as session:
            self.assertEqual(session['files'], [])
            self.assertTrue(session['error'])

    def test_failed_crop_write_cleans_up_partial_images(self):
        with patch.object(web.cv2, 'imwrite', return_value=False):
            with self.assertLogs(web.app.logger, level='ERROR'):
                response = self.upload()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_spy_again_handles_a_missing_original(self):
        self.upload()
        self.process.side_effect = FileNotFoundError('Original image is missing')
        with self.assertLogs(web.app.logger, level='ERROR'):
            response = self.client.post('/again', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_spy_again_without_a_round_redirects(self):
        response = self.client.post('/again', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.process.assert_not_called()


if __name__ == '__main__':
    unittest.main()
