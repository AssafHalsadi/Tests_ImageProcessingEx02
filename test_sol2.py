import unittest

from pathlib import Path

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


def _generate_images(names):
    images = []
    for name in names:
        images.append((read_image(os.path.join(os.path.abspath(r'external'), "{}.jpg".format(name)), 1), name))
    return images


pdf_ratio = 1.25
smallest_ratio = 0.26
largest_ratio = 3.9
double_ratio = 2
half_ratio = 0.5
same = 1
ratios = [pdf_ratio, smallest_ratio, largest_ratio, double_ratio, half_ratio, same]

arr_pdf = (np.arange(1000), "arr_pdf")
arr_large_zeros = (np.zeros_like(arr_pdf), "arr_large_zeros")
arr_large_ones = (np.ones_like(arr_pdf), "arr_large_ones")
arr_normal = (np.array([1, 2, 3]), "arr_normal")
arr_same_val = (np.array([1, 1, 1]), "arr_same_val")
arr_zero_vals = (np.array([0, 0, 0]), "arr_zero_vals")
arr_single_cell = (np.array([1]), "arr_single_cell")
arr_single_zero = (np.array([0]), "arr_single_zero")
arr_empty = (np.array([]), "arr_empty")

test_arrs = [arr_pdf, arr_normal, arr_same_val, arr_zero_vals, arr_single_cell, arr_single_zero, arr_empty,
             arr_large_ones, arr_large_zeros]


# ================================ helper functions ================================


def _does_contain(function, statements):
    nodes = ast.walk(ast.parse(inspect.getsource(function)))
    return any(isinstance(node, statements) for node in nodes)


def _uses_loop(function):
    loop_statements = ast.For, ast.While, ast.AsyncFor
    return _does_contain(function, loop_statements)


def _has_return(function):
    return _does_contain(function, ast.Return)


# ================================ unittest class ================================


class TestEx2(unittest.TestCase):
    aria_path = os.path.abspath(r'external/aria_4kHz.wav')
    monkey_path = os.path.abspath(r'external/monkey.jpg')

    # ================================ setup/teardown functions ================================

    @classmethod
    def setUpClass(cls):
        cls.aria_rate, cls.aria_data = wavfile.read(cls.aria_path)
        cls.reshaped_aria = cls.aria_data.reshape(cls.aria_data.shape[0], 1)
        cls.monkey_color = read_image(cls.monkey_path, 2)
        cls.monkey_grayscale = read_image(cls.monkey_path, 1)
        cls.images = _generate_images(['monkey', 'city', 'trees', 'view', 'waterfall', 'woman'])

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # ================================ Part I Tests ================================

    # -------------------------------- helper functions  --------------------------------

    def _module_test(self, func, func_in, system_func, out_info, sys_in, signature, no_loops, is_sound):
        func_name = str(func.__name__)
        sys_func_name = str(system_func.__name__)
        output_shape = str(out_info[0])
        output_type = out_info[1]

        # Check no loops if needed
        if no_loops:
            self.assertEqual(_uses_loop(func), False,
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

    # -------------------------------- 1.1 test  --------------------------------

    # todo: Check return types
    def test_DFT_IDFT_1D(self):
        # ==== Test DFT ====
        pass
        # dft_out, fft_out = self._module_test(func=sol.DFT, func_in=(self.reshaped_aria,), system_func=np.fft.fft,
        #                                      out_info=((9600, 1), np.dtype('complex128')), sys_in=(self.aria_data,),
        #                                      signature=r'(signal)',
        #                                      no_loops=True, is_sound=True)
        #
        # # ==== Test IDFT ====
        #
        # self._module_test(func=sol.IDFT, func_in=(dft_out,), system_func=np.fft.ifft,
        #                   out_info=((9600, 1), np.dtype('complex128')), sys_in=(fft_out,),
        #                   signature=r'(fourier_signal)',
        #                   no_loops=True, is_sound=True)

    # -------------------------------- 1.2 test  --------------------------------

    def test_DFT2_IDFT2(self):
        pass
        # # ==== Test DFT2 ====
        #
        # dft2_out, fft2_out = self._module_test(func=sol.DFT2, func_in=(self.monkey_grayscale,), system_func=np.fft.fft2,
        #                                        out_info=((500, 418), np.dtype('complex128')),
        #                                        sys_in=(self.monkey_grayscale,), signature=r'(image)',
        #                                        no_loops=False, is_sound=False)
        # # ==== Test IDFT2 ====
        # # todo: check if out_type here is float64 or complex128
        # self._module_test(func=sol.IDFT2, func_in=(dft2_out,), system_func=np.fft.ifft2,
        #                   out_info=((500, 418), np.dtype('float64')), sys_in=(fft2_out,), signature=r'(fourier_image)',
        #                   no_loops=False, is_sound=False)

    # ================================ Part II Tests ================================

    # -------------------------------- helper functions  --------------------------------

    def _test_speedup_module(self, func, ratio, first_arg, acc):
        orig_time = len(self.aria_data) / self.aria_rate
        new_time = orig_time / ratio

        if func.__name__ == "resize_spectrogram":
            sol_rate = self.aria_rate
            sol_data = func(first_arg, np.float64(ratio))
        else:
            func(first_arg, np.float64(ratio))
            sol_rate, sol_data = wavfile.read(os.path.abspath('{}.wav'.format(func.__name__)))

        if func.__name__ == "change_rate":
            self.assertIsNone(np.testing.assert_array_equal(self.aria_data, sol_data,
                                                            err_msg=r'wav file data should not be changed by "change_rate" function'))
        else:
            self.assertEqual(self.aria_rate, sol_rate,
                             msg='"{}" should not change the sample rate'.format(func.__name__))
        # print("orig time : {}\nratio : {}\nnew time : {}\nsol new time : {}\n=====\n".format(orig_time, ratio, new_time, len(sol_data) / sol_rate))
        self.assertAlmostEqual(new_time, len(sol_data) / sol_rate, delta=acc,
                               msg='Old duration was {} seconds, ratio was {}, new duration should be {} seconds. Check your calculations.'.format(
                                   orig_time, ratio, new_time))

    # -------------------------------- 2.1 test --------------------------------

    def test_change_rate(self):
        pass
        # ==== Structure testing ====
        #
        # self.assertEqual(str(inspect.signature(sol.change_rate)), r'(filename, ratio)')
        #
        # self.assertEqual(_has_return(sol.change_rate), False,
        #                  msg=r'"change_rate" function should not have a return statement')
        #
        # # ==== Testing speed over different ratios ====
        #
        # for ratio in ratios:
        #     self._test_speedup_module(sol.change_rate, ratio, self.aria_path, 1.e-3)

    # -------------------------------- 2.2 tests --------------------------------

    def _test_resize_helper(self, arr, name):

        for ratio in ratios:
            result = sol.resize(arr, ratio)

            self.assertEqual(1, len(result.shape), msg=r'"change_samples" returned array should be 1D')

            self.assertTrue(result.dtype in [np.dtype('float64'), np.dtype('complex128')],
                            msg=r'"change_samples" returned array should be of dtype np.float64 or np.complex128')

            self.assertEqual(result.shape[0], arr.shape[0] // ratio,
                             msg='"change_samples" returned array\'s length is wrong on "' + name + '" array and "' + str(
                                 ratio) + '" ratio.')

    def test_resize(self):
        pass
        # # todo: comment - DOES NOT TEST *HOW* YOU RESIZED THE ARRAY, ONLY THAT IT IS RESIZED TO THE RIGHT LENGTH
        # # ==== Structure testing ====
        #
        # self.assertEqual(str(inspect.signature(sol.resize)), r'(data, ratio)')
        #
        # # todo: check if should not have loops
        # # todo: check if an empty array might be sent as input
        #
        # for arr in test_arrs:
        #     self._test_resize_helper(arr[0].astype(np.float64), arr[1])

    def test_change_samples(self):
        pass
        # ==== Structure testing ====
        #
        # self.assertEqual(str(inspect.signature(sol.change_samples)), r'(filename, ratio)')
        #
        # # ==== Testing speed over different ratios ====
        #
        # for ratio in ratios:
        #     if ratio >= 0.5:
        #         self._test_speedup_module(sol.change_samples, ratio, self.aria_path, 1.e-3)

    # -------------------------------- 2.3 tests --------------------------------

    # todo: SPECIFY THAT BECAUSE STFT AND ISTFT ARE NOT PRECISE, RESULTS MAY VARY OR FLAT OUT BE WRONG
    def test_resize_spectogram(self):
        # self.assertTrue(False)
        # pass
        ==== Structure testing ====

        self.assertEqual(str(inspect.signature(sol.resize_spectrogram)), r'(data, ratio)')

        # ==== Testing speed over different ratios ====

        for ratio in ratios:
            if ratio >= 1:
                self._test_speedup_module(sol.resize_spectrogram, ratio, self.aria_data, 1.e-1)
            else:
                self._test_speedup_module(sol.resize_spectrogram, ratio, self.aria_data, 5.e-1)
    #
    -------------------------------- 2.4 tests --------------------------------
    #
    # todo: SPECIFY THAT BECAUSE STFT AND ISTFT ARE NOT PRECISE, RESULTS MAY VARY OR FLAT OUT BE WRONG
    def test_resize_vocoder(self):
        # ==== Structure testing ====

        self.assertEqual(str(inspect.signature(sol.resize_spectrogram)), r'(data, ratio)',
                         msg='"resize_spectrogram"\'s signature is not as requested.')

        # ==== Testing speed over different ratios ====

        for ratio in ratios:
            if ratio >= 1:
                self._test_speedup_module(sol.resize_spectrogram, ratio, self.aria_data, 1.e-1)
            else:
                self._test_speedup_module(sol.resize_spectrogram, ratio, self.aria_data, 5.e-1)

    # ================================ Part III Tests ================================


    def _test_der_module(self, func, name):
        ==== Structure testing ====
        self.assertEqual(str(inspect.signature(func)), r'(im)',
                         msg='"{}"\'s signature is not as requested.'.format(name))



        for im in self.images:
            rel_path = r'output_compare/{}_mag.csv' if (name == 'conv_der') else r'output_compare/{}_mag.csv'
            np.savetxt(os.path.abspath(rel_path.format(im[1])), sol.conv_der(im[0]), delimiter=",")
            saved_mag = np.loadtxt(os.path.abspath(rel_path.format(im[1])), np.float64, delimiter=",")

            mag = func(im[0])

            self.assertEqual(im[0].shape, mag.shape,
                             msg=r'Derivative magnitude matrix\'s shape should be equal to the original image.')

            self.assertIsNone(np.testing.assert_array_equal(mag, saved_mag))

    -------------------------------- 3.1  --------------------------------

    # todo: SPECIFY ITS COMPARED TO MY OUTPUT
    def test_conv_der(self):
        self._test_der_module(sol.conv_der, sol.conv_der.__name__)

    -------------------------------- 3.2  --------------------------------

    # todo: SPECIFY ITS COMPARED TO MY OUTPUT
    def test_fourier_der(self):
        self._test_der_module(sol.fourier_der, sol.fourier_der.__name__)


if __name__ == '__main__':
    unittest.main()