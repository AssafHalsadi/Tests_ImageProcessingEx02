import unittest
import sol2 as sol
from imageio import imread
from skimage.color import rgb2gray
import numpy as np
from scipy.io import wavfile
import os
import inspect
import ast


def read_image(filename, representation):
    """
    Receives an image file and converts it into one of two given representations.
    :param filename: The file name of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining wether the output
    should be a grayscale image (1) or an RGB image (2). If the input image is grayscale,
    we won't call it with representation = 2.
    :return: An image, represented by a matrix of type (np.float64) with intensities
    normalized to the range [0,1].
    """
    assert representation in [1, 2]

    # reads the image
    im = imread(filename)
    if representation == 1:  # If the user specified they need grayscale image,
        if len(im.shape) == 3:  # AND the image is not grayscale yet
            im = rgb2gray(im)  # convert to grayscale (**Assuming its RGB and not a different format**)

    im_float = im.astype(np.float64)  # Convert the image type to one we can work with.

    if im_float.max() > 1:  # If image values are out of bound, normalize them.
        im_float = im_float / 255

    return im_float


class TestEx2(unittest.TestCase):
    aria_path = os.path.abspath(r'external/aria_4kHz.wav')
    monkey_path = os.path.abspath(r'external/monkey.jpg')

    def uses_loop(self, function):
        loop_statements = ast.For, ast.While, ast.AsyncFor

        nodes = ast.walk(ast.parse(inspect.getsource(function)))
        return any(isinstance(node, loop_statements) for node in nodes)

    @classmethod
    def setUpClass(cls):
        cls.aria_rate, cls.aria_data = wavfile.read(cls.aria_path)
        cls.reshaped_aria = cls.aria_data.reshape(cls.aria_data.shape[0], 1)
        cls.monkey_color = read_image(cls.monkey_path, 2)
        cls.monkey_grayscale = read_image(cls.monkey_path, 1)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _module_test(self, func, func_in, system_func, out_info, sys_in, signature, no_loops, is_sound):
        func_name = str(func.__name__)
        sys_func_name = str(system_func.__name__)
        output_shape = str(out_info[0])
        output_type = out_info[1]

        # Check no loops if needed
        if no_loops:
            self.assertEqual(self.uses_loop(func), False,
                             msg="Your %s implementation should not contain loops" % func_name)

        # Checks the signature of the function
        self.assertEqual(str(inspect.signature(func)), signature)

        # init values
        func_out = func(*func_in)
        sys_out = system_func(*sys_in)

        # Checks shape of output
        self.assertEqual(func_out.shape, out_info[0], msg=func_name + ' returned shape should be ' + output_shape)

        # Checks type of output
        self.assertEqual(func_out.dtype, output_type, msg=func_name + ' returned type should be ' + str(output_type))

        # Compares values of output to the values of the built in function, up to 5 points after the decimal
        if is_sound:
            func_out = func_out.flatten()
        self.assertIsNone(np.testing.assert_array_almost_equal(func_out, sys_out, decimal=5,
                                                               err_msg=r"Output is too different from %s implementation, please check your code again" % sys_func_name))

        return func_out, sys_out

    def test_DFT_IDFT_1D(self):
        # ==== Test DFT ====

        dft_out, fft_out = self._module_test(func=sol.DFT, func_in=(self.reshaped_aria,), system_func=np.fft.fft,
                                             out_info=((9600, 1), np.dtype('complex128')), sys_in=(self.aria_data,),
                                             signature=r'(signal)',
                                             no_loops=True, is_sound=True)

        # ==== Test IDFT ====

        self._module_test(func=sol.IDFT, func_in=(dft_out,), system_func=np.fft.ifft,
                          out_info=((9600, 1), np.dtype('complex128')), sys_in=(fft_out,), signature=r'(fourier_signal)',
                          no_loops=True, is_sound=True)

    def test_DFT2_IDFT2(self):
        # ==== Test DFT2 ====

        dft2_out, fft2_out = self._module_test(func=sol.DFT2, func_in=(self.monkey_grayscale,), system_func=np.fft.fft2,
                                               out_info=((500, 418), np.dtype('complex128')),
                                               sys_in=(self.monkey_grayscale, ), signature=r'(image)',
                                               no_loops=False, is_sound=False)
        # ==== Test IDFT2 ====
        # todo: check if out_type here is float64 or complex128
        self._module_test(func=sol.IDFT2, func_in=(dft2_out,), system_func=np.fft.ifft2,
                          out_info=((500, 418), np.dtype('float64')), sys_in=(fft2_out,), signature=r'(fourier_image)',
                          no_loops=False, is_sound=False)


if __name__ == '__main__':
    unittest.main()
