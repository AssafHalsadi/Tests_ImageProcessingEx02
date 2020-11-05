# HUJI Image Processing course (67829) Ex2 Tests 2020/2021<a name="TOP"></a>
[![Build Status](https://img.shields.io/badge/build%20version-1.0-green)](https://github.com/AssafHalsadi/Tests_ImageProcessingEx02)
![](../readme_assets/cover_image.jpg)

Testing suite for the second exercise of Image Processing course (67829) at HUJI. The suite includes basic tests for all of the exercises functions that checks the API, the return values, usage of loops and some functionality. In this README I will go over [requirements](#REQ), a [guide](#HOWTO) of how to use the tests, each tests coverage and what it means to pass it.


## :warning: DISCLAIMER :warning:
Passing these tests DOES NOT guaranty you will get a good grade in any way, as they are not moderated by the course's staff.
I will try and make it as clear as possible as to which extent the tests cover the exercise, but i felt the disclaimer was needed in any case.

## :books: Documentation
[![](https://user-images.githubusercontent.com/4301109/70686099-3855f780-1c79-11ea-8141-899e39459da2.png)](https://assafhalsadi.github.io/Tests_ImageProcessingEx02/. )

:arrow_up: _Easy navigation through this README file can be found here_ :arrow_up:

## Collaborators
[Assaf Halsadi](https://github.com/AssafHalsadi) :israel:

## Contact Info
If you find any mistakes, or have any questions - please contact me through the course's Slack [![Slack](https://img.shields.io/badge/HUJI_IMPR_20%20slack-join-green?logo=slack&labelColor=4A154B)](https://join.slack.com/t/huji-impr-20/shared_invite/zt-i5z2lgja-vs8c6RptH8t2_jou~Wvhuw)

Or at the courses forum at the relevant post [![forum](https://img.shields.io/badge/moodle%20forum-goto-orange?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAXCAMAAAAm/38fAAABdFBMVEUNDg4ODw8PEA8PEBAQEA8QEBEQEhMREhEREhISEhETEhATEhEUEhAUFBQVFBEWFxcXFBEXFxgZFRAZFREaFhEcFREcHh4hGBIhHh0iISEkGhElHBQlHRglJCUmJCUmJCYnHhUnJCQoHRQoJSMpHhQpJygqHRIqHhUrJCEsHRQtJSEtKSsxIxQxLi8zLCo0MjQ1JBQ2JhU2KyU3MzE3NDU4IhY5IhU5JBQ5KBc9P0E/JhRAKRVBKxVDQEFDQkVEKBlEPDlEPDtJMBpMTE9PKxlQS0tRRD9TLhpUPzBUSEJjWlhmQh9oXlppQyJuRyFwPhlyV0Z2QRp2QRuPUiCPa1OWViKYWCOfXy2qaSmsZSa0bSq1biq6aye8cCi8cyq/bCjBdCvFcCjFcSjHcijLfC3NbifReyvUdCvWeyvZdCnZfivadSnffyvfgivigizreSftiy7viyvwfSnwhCryhi3zfSjzfij1hyv1iSz2jiz2jyz2jy3O+s8EAAAAtUlEQVR42mMY7ICVg5cXQ5BZUFJOUVGWH02YXdnBzVFXW56NS1pBAElcwtPHy9XWRNXINzIjK9abDy4hoyVm6WynrxdeUpGXW1ngB5cQF2XwCAkw1ogvz3GySKuKg0vYmzIEFaWrM4SVJ4twRlelwCXcXRgCy7JVGELLk4Q5o6oREppmDIGlmSCJRCEUCSZuZAmQUQjgn5+qxBBcHCPEGVGYgCwhZWXOw6BmbcDCqGNjOPCxAwBHvSY/OMtgWQAAAABJRU5ErkJggg==&labelColor=4A154B)](https://moodle2.cs.huji.ac.il/nu20/course/view.php?id=67829)

## Requirements<a name="REQ"></a>
To run the tests you will only need the following things:

[![python](https://img.shields.io/badge/python-3-blue.svg?logo=python&labelColor=yellow)](https://www.python.org/downloads/)
[![platform](https://img.shields.io/badge/platform-osx%2Flinux%2Fwindows-green.svg)](https://github.com/AssafHalsadi/Tests_ImageProcessingEx02)
[![file](https://img.shields.io/badge/file-ex2__helper.py-red)](https://moodle2.cs.huji.ac.il/nu20/course/view.php?id=67829)
[![file](https://img.shields.io/badge/file-sol2.py-red)](https://moodle2.cs.huji.ac.il/nu20/course/view.php?id=67829)

## How to - running the tests<a name="HOWTO"></a>
### Setup
1. Clone this repository into a _"tests"_ folder within your project's root folder:
    * Open a new folder named _"tests"_ in your project's root folder.
    * Open a command prompt on your computer, I will use cmd as an example on windows:
    <img src="../readme_assets/02.png" width="48">
    
    * Go to the _tests_ folder using the `cd` command like so : `_cd [path_to_project]/tests` (change [path_to_project] with the path to your project):
     
    ![cd](../readme_assets/03.png)
    * Go to the [top of the page](#TOP), there you should copy the git link: 
    
    [![copy link](../readme_assets/01.png)](#TOP)
    * Type `git clone *copy here*` :
    
     ![clone](../readme_assets/04.png)
    * You might be prompted to enter your [CSE user credentials](https://wiki.cs.huji.ac.il/wiki/Password_and_OTP#OTP_and_UNIX_passwords)
2. Unpack the _output_compare.rar_ located in the _output_compare_ folder.
 ![unpack](../readme_assets/05.png)
3. Copy both _sol2.py_, _ex2_helper.py_ and any other files needed for your implementation to the _tests_ folder. 
3. That is it, no need for complicated voodoo. :smile:

### Usage
There are two main ways to run the tests, via the Textual Interface or via pycharm's built in unittest support.
#### Textual Interface
1. Go to the project's folder.
2. Go to the _tests_ sub-folder.
3. Double click on _'RunMe.bat'_:
 ![runme](../readme_assets/06.png)
If everything went according to plan, you should see a cmd window opens, and after a while the tests will start running.

**Remark**: Some of the tests take A LOT of time to complete. To run only SOME of the tests open the _runner.py_ and scroll down to this function:
![testList](../readme_assets/07.png)
Flip the comment like so:
![testList02](../readme_assets/08.png)
and delete the names of the tests you don't want to run.
#### Pycharm
1. Go to _test _ sol2.py_ file, located in the "tests" folder.
2. To run all of the tests, scroll down to the TestSuite start and click the green "play" button :
![runAll](../readme_assets/09.png)
3. To run an individual test, scroll down to the test's function and click on the green "play" button beside it :
![runSingle](../readme_assets/10.png)
You can identify tests by the face they all start with `def test_...`






