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


def _generate_images(names):
    """
    Generates a list of images from a list of image names.
    :param names: List of strings.
    :return: A list of grayscale images.
    """
    images = []
    for name in names:
        images.append((read_image(os.path.join(os.path.abspath(r'external'), f"{name}.jpg"), 1), name))
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
    """
    Checks if a function implementation contains any usage of given tokens.
    :param function: The function to check in.
    :param statements: The statement tokens to find.
    :return: True if there is an instance of the statements in the function implementation, False otherwise.
    """
    nodes = ast.walk(ast.parse(inspect.getsource(function)))
    return any(isinstance(node, statements) for node in nodes)


def _uses_loop(function):
    """
    Checks if a function uses top level loops.
    :param function: The function to check in.
    :return: True if it contains loops, False otherwise.
    """
    loop_statements = ast.For, ast.While, ast.AsyncFor
    return _does_contain(function, loop_statements)


def _has_return(function):
    """
    Checks if a function contains a return statement.
    :param function: The function to check in.
    :return: True if it contains a return statement, False otherwise.
    """
    return _does_contain(function, ast.Return)


# ================================ unittest class ================================


class TestEx2(unittest.TestCase):
    """
    The unittest testing suite.
    """
    # Path to example wav supplied by the course's staff.
    aria_path = os.path.abspath(r'external/aria_4kHz.wav')
    # Path to example jpg supplied by the course's staff.
    monkey_path = os.path.abspath(r'external/monkey.jpg')

    # ================================ setup/teardown functions ================================

    @classmethod
    def setUpClass(cls):
        """
        Generates all necessary data for tests, runs before all other tests.
        :return: -
        """
        # Generates the wav.
        cls.aria_rate, cls.aria_data = wavfile.read(cls.aria_path)
        cls.reshaped_aria = cls.aria_data.reshape(cls.aria_data.shape[0], 1)

        # Generates images.
        cls.monkey_grayscale = read_image(cls.monkey_path, 1)
        cls.reshaped_monkey_grayscale = cls.monkey_grayscale.reshape(cls.monkey_grayscale.shape[0],
                                                                     cls.monkey_grayscale.shape[1], 1)
        cls.images = _generate_images(['monkey', 'city', 'trees', 'view', 'waterfall', 'woman'])

    # ================================ Part I Tests ================================

    # -------------------------------- helper functions  --------------------------------

    def _test_fourier_module(self, func, func_in, system_func, out_info, sys_in, signature, no_loops, is_sound):
        """
        Test module for DFT, IDFT, DFT2 and IDFT2.
        :param func: The function to test.
        :param func_in: (tuple) Input to function.
        :param system_func: The built in function to compare to.
        :param out_info: Information about the expected function's output, includes shape and dtype in a tuple.
        :param sys_in: (tuple) Input to built in function.
        :param signature: Expected signature of the function.
        :param no_loops: (Boolean) Should there be no loops in the implementation?
        :param is_sound: (Boolean) Is the input wav or image.
        :return: Output of both func and system_func
        """
        func_name = str(func.__name__)
        sys_func_name = str(system_func.__name__)
        output_shape = str(out_info[0])
        output_type = out_info[1]

        # Check no loops if needed
        if no_loops:
            self.assertEqual(False, _uses_loop(func),
                             msg=f"Your {func_name} implementation should not contain loops")

        # Checks the signature of the function
        self.assertEqual(signature, str(inspect.signature(func)))

        # init values
        func_out = func(*func_in)
        sys_out = system_func(*sys_in)

        # Checks shape of output
        self.assertEqual(out_info[0], func_out.shape, msg=f'{func_name} returned shape should be {output_shape}')

        # Checks type of output
        self.assertEqual(output_type, func_out.dtype, msg=f'{func_name} returned type should be {str(output_type)}')

        # Compares values of output to the values of the built in function, up to 5 points after the decimal
        test_out = func_out
        if is_sound:
            test_out = func_out.flatten()
        else:
            test_out = test_out.reshape(test_out.shape[0], test_out.shape[1])
        self.assertIsNone(np.testing.assert_array_almost_equal(sys_out, test_out, decimal=5,
                                                               err_msg=f"Output is too different from {sys_func_name} implementation, please check your code again"))

        return func_out, sys_out

    # -------------------------------- 1.1 test  --------------------------------

    # todo: Check return types
    def test_DFT_IDFT_1D(self):
        """
        Tests both DFT and IDFT functions by comparing them to the built in np.fft.___ functions.
        Allows 1.e-5 difference.
        :return: -
        """
        # ==== Test DFT ====
        dft_out, fft_out = self._test_fourier_module(func=sol.DFT, func_in=(self.reshaped_aria,),
                                                     system_func=np.fft.fft,
                                                     out_info=(self.reshaped_aria.shape, np.dtype('complex128')),
                                                     sys_in=(self.aria_data,),
                                                     signature=r'(signal)',
                                                     no_loops=True, is_sound=True)

        dft_out_normal, fft_out_normal = self._test_fourier_module(func=sol.DFT, func_in=(self.aria_data,),
                                                     system_func=np.fft.fft,
                                                     out_info=(self.aria_data.shape, np.dtype('complex128')),
                                                     sys_in=(self.aria_data,),
                                                     signature=r'(signal)',
                                                     no_loops=True, is_sound=True)

        # ==== Test IDFT ====

        self._test_fourier_module(func=sol.IDFT, func_in=(dft_out,), system_func=np.fft.ifft,
                                  out_info=(self.reshaped_aria.shape, np.dtype('complex128')), sys_in=(fft_out,),
                                  signature=r'(fourier_signal)',
                                  no_loops=True, is_sound=True)

        self._test_fourier_module(func=sol.IDFT, func_in=(dft_out_normal,), system_func=np.fft.ifft,
                                  out_info=(self.aria_data.shape, np.dtype('complex128')), sys_in=(fft_out_normal,),
                                  signature=r'(fourier_signal)',
                                  no_loops=True, is_sound=True)



    # -------------------------------- 1.2 test  --------------------------------

    def test_DFT2_IDFT2(self):
        """
        Tests both DFT2 and IDFT2 functions by comparing them to the built in np.fft.___ functions.
        Allows 1.e-5 difference.
        :return:
        """
        # ==== Test DFT2 ====

        dft2_out, fft2_out = self._test_fourier_module(func=sol.DFT2, func_in=(self.reshaped_monkey_grayscale,),
                                                       system_func=np.fft.fft2,
                                                       out_info=(self.reshaped_monkey_grayscale.shape, np.dtype('complex128')),
                                                       sys_in=(self.monkey_grayscale,), signature=r'(image)',
                                                       no_loops=False, is_sound=False)

        dft2_out_normal, fft2_out_normal = self._test_fourier_module(func=sol.DFT2, func_in=(self.monkey_grayscale,),
                                                       system_func=np.fft.fft2,
                                                       out_info=(self.monkey_grayscale.shape, np.dtype('complex128')),
                                                       sys_in=(self.monkey_grayscale,), signature=r'(image)',
                                                       no_loops=False, is_sound=False)

        # ==== Test IDFT2 ====

        self._test_fourier_module(func=sol.IDFT2, func_in=(dft2_out,), system_func=np.fft.ifft2,
                                  out_info=(self.reshaped_monkey_grayscale.shape, np.dtype('complex128')), sys_in=(fft2_out,),
                                  signature=r'(fourier_image)',
                                  no_loops=False, is_sound=False)

        self._test_fourier_module(func=sol.IDFT2, func_in=(dft2_out_normal,), system_func=np.fft.ifft2,
                                  out_info=(self.monkey_grayscale.shape, np.dtype('complex128')), sys_in=(fft2_out_normal,),
                                  signature=r'(fourier_image)',
                                  no_loops=False, is_sound=False)

    # ================================ Part II Tests ================================

    # -------------------------------- helper functions  --------------------------------

    def _test_speedup_module(self, func, ratio, first_arg, acc):
        """
        Test module for change_rate, change_samples, resize_spectrogram and resize_vocoder.
        Checks mainly that the outputted wav file's speed is the original speed/ratio.
        In case of change_rate, checks the data did not change.
        In case of any other function, checks the rate did not change.
        :param func: The function to test.
        :param ratio: The ratio of change in speed.
        :param first_arg: The first argument to the function (only argument that differs between them).
        :param acc: The accuracy expected of the tested function.
        :return: -
        """
        orig_time = len(self.aria_data) / self.aria_rate
        new_time = orig_time / ratio

        if func.__name__ == "resize_spectrogram":
            sol_rate = self.aria_rate
            sol_data = func(first_arg, np.float64(ratio))
        elif func.__name__ == "change_samples":
            out_data = func(first_arg, np.float64(ratio))
            self.assertEqual(np.dtype("float64"), out_data.dtype, msg=r'change_samples returned dtype should be float64')
            self.assertEqual(1, len(out_data.shape), msg=r'change_samples output should be 1D')
            sol_rate, sol_data = wavfile.read(os.path.abspath(f'{func.__name__}.wav'))
        else:
            func(first_arg, np.float64(ratio))
            sol_rate, sol_data = wavfile.read(os.path.abspath(f'{func.__name__}.wav'))

        if func.__name__ == "change_rate":
            self.assertIsNone(np.testing.assert_array_equal(self.aria_data, sol_data,
                                                            err_msg=r'wav file data should not be changed by "change_rate" function'))
        else:
            self.assertEqual(self.aria_rate, sol_rate, msg=f'"{func.__name__}" should not change the sample rate')

        # print("orig time : {}\nratio : {}\nnew time : {}\nsol new time : {}\n=====\n".format(orig_time, ratio, new_time, len(sol_data) / sol_rate))
        self.assertAlmostEqual(new_time, len(sol_data) / sol_rate, delta=acc,
                               msg=f'Old duration was {str(orig_time)} seconds, ratio was {str(ratio)}, new duration should be {str(new_time)} seconds. Check your calculations.')

    # -------------------------------- 2.1 test --------------------------------

    def test_change_rate(self):
        """
        Tests the change rate function by comparing the outputted wav speed to the speed its supposed to be in
        and also makes sure the data did not change.
        :return: -
        """
        # ==== Structure testing ====

        # Tests signature
        self.assertEqual(r'(filename, ratio)', str(inspect.signature(sol.change_rate)))

        # Makes sure the function does not return anything
        self.assertEqual(False, _has_return(sol.change_rate),
                         msg=r'"change_rate" function should not have a return statement')

        # ==== Testing speed over different ratios ====

        for ratio in ratios:
            self._test_speedup_module(sol.change_rate, ratio, self.aria_path, 1.e-3)

    # -------------------------------- 2.2 tests --------------------------------

    def _test_resize_helper(self, arr, name):
        """
        Helper function that tests the "resize" function functionality on a specific given array, for all ratios.
        :param arr: Array to test on.
        :param name: Name of the array test.
        :return: -
        """
        for ratio in ratios:
            result = sol.resize(arr, ratio)

            # makes sure the returned array is 1D
            self.assertEqual(1, len(result.shape), msg=r'"change_samples" returned array should be 1D')

            # makes sure the returned dtype is correct according to pdf
            self.assertTrue(result.dtype in [np.dtype('float64'), np.dtype('complex128')],
                            msg=r'"change_samples" returned array should be of dtype np.float64 or np.complex128')

            # Compares
            self.assertEqual(arr.shape[0] // ratio, result.shape[0],
                             msg=f'"change_samples" returned array\'s length is wrong on {name} array and {str(ratio)} ratio.')

    def test_resize(self):
        """
        Tests resize function by checking the outputted arrays have the correct length in correspondance to the given
        ratio. DOES NOT test how the array was resized.
        :return: -
        """
        # ==== Structure testing ====

        self.assertEqual(r'(data, ratio)', str(inspect.signature(sol.resize)))

        # todo: check if should not have loops
        # todo: check if an empty array might be sent as input

        for arr in test_arrs:
            self._test_resize_helper(arr[0].astype(np.float64), arr[1])

    def test_change_samples(self):
        """
        Tests the "change_samples" function by using the speed test module.
        :return: -
        """
        # ==== Structure testing ====

        self.assertEqual(r'(filename, ratio)', str(inspect.signature(sol.change_samples)))

        # The function used to not have a returned statement, it changed
        # self.assertEqual(False, _has_return(sol.change_samples),
        #                  msg=r'"change_samples" function should not have a return statement')

        # ==== Testing speed over different ratios ====

        for ratio in ratios:
            if ratio >= 0.5:
                self._test_speedup_module(sol.change_samples, ratio, self.aria_path, 1.e-3)

    # -------------------------------- 2.3 tests --------------------------------

    # todo: SPECIFY THAT BECAUSE STFT AND ISTFT ARE NOT PRECISE, RESULTS MAY VARY OR FLAT OUT BE WRONG
    def test_resize_spectrogram(self):
        """                                                                
        Tests the "resize_spectrogram" function by using the speed test module.
        :return: -                                                         
        """
        # ==== Structure testing ====

        self.assertEqual(r'(data, ratio)', str(inspect.signature(sol.resize_spectrogram)))

        # ==== Testing speed over different ratios ====

        for ratio in ratios:
            if ratio >= 1:
                self._test_speedup_module(sol.resize_spectrogram, ratio, self.aria_data, 1.e-1)
            else:
                self._test_speedup_module(sol.resize_spectrogram, ratio, self.aria_data, 5.e-1)

    # -------------------------------- 2.4 tests --------------------------------

    # todo: SPECIFY THAT BECAUSE STFT AND ISTFT ARE NOT PRECISE, RESULTS MAY VARY OR FLAT OUT BE WRONG
    def test_resize_vocoder(self):
        """
        Tests the "resize_vocoder" function by using the speed test module.
        :return: -
        """
        # ==== Structure testing ====

        self.assertEqual(r'(data, ratio)', str(inspect.signature(sol.resize_vocoder)),
                         msg='"resize_spectrogram"\'s signature is not as requested.')

        # ==== Testing speed over different ratios ====

        for ratio in ratios:
            if ratio >= 1:
                self._test_speedup_module(sol.resize_spectrogram, ratio, self.aria_data, 1.e-1)
            else:
                self._test_speedup_module(sol.resize_spectrogram, ratio, self.aria_data, 5.e-1)

##############################################################################
################################# DEPRECATED #################################
##############################################################################

    # ================================ Part III Tests ================================

    def _test_der_module(self, func, name):
        """
        Testing module for conv_der and fourier_der functions.
        Compares their output on multiple inputs TO MY OUTPUT.
        :param func: The function to test.
        :param name: Name of the function being tested.
        :return: -
        """
        # ==== Structure testing ====
        self.assertEqual(r'(im)', str(inspect.signature(func)),
                         msg=f'"{name}"\'s signature is not as requested.')

        for im in self.images:
            # Chooses path according to the function that is being tested.
            rel_path = r'output_compare/{}_mag.csv' if (name == 'conv_der') else r'output_compare/{}_fourier_mag.csv'

            # Code that saved my output
            # np.savetxt(os.path.abspath(rel_path.format(im[1])), func(im[0]), delimiter=",")

            # Retrieves my output
            saved_mag = np.loadtxt(os.path.abspath(rel_path.format(im[1])), np.float64, delimiter=",")

            # Gets tested output
            mag = func(im[0])

            # Checks the output's shape.
            self.assertEqual(im[0].shape, mag.shape,
                             msg=r'Derivative magnitude matrix\'s shape should be equal to the original image.')

            # Compares outputs
            if name == 'conv_der':
                self.assertIsNone(np.testing.assert_array_equal(saved_mag, mag))
            else:
                self.assertIsNone(np.testing.assert_array_almost_equal(saved_mag, mag, decimal=3))

# # -------------------------------- 3.1  --------------------------------
#
# def test_conv_der(self):
#     """
#     Tests the "conv_der" function by using the derivative testing module.
#     :return: -
#     """
#     self._test_der_module(sol.conv_der, sol.conv_der.__name__)
#
# # -------------------------------- 3.2  --------------------------------
#
# def test_fourier_der(self):
#     """
#     Tests the "fourier_der" function by using the derivative testing module.
#     :return: -
#     """
#     self._test_der_module(sol.fourier_der, sol.fourier_der.__name__)

##############################################################################
################################# DEPRECATED #################################
##############################################################################


if __name__ == '__main__':
    unittest.main()
