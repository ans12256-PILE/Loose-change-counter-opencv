# Loose-change-counter-opencv
Python opencv excercise to mess with image recognition

January 16, 2021; Alexey N Smirnov
capstone project for Python refresh class - loved it 🤓:
https://www.udemy.com/course/the-complete-python-programmer-bootcamp/

A lot of things came together - finally learning python environment(could not install opencv otherwise in anaconda, but then could install R-studio as well), github, python 3.x vs Python 2.x refresh/update, and of course opencv basics.

Extensive online consultation was used in completion of the project, with proper credits given in the code comments.
Some of the algorythms are different compared to instructor's solution for example
1) Edge detection parameters are more accurate
2) Relative sizes of the British coins were first researched (refer attached figure), and used in the algorythm
3) Color, rather than brightness was used in detection algorythm (refer attached excerpt from Excel file)
4) Full coin circular mask was used before performing color analysis (a more elegant+-R/2 square would be simpler)
5) No coin overlap was analyzed, but then was deemed an unnecessary complication. Even human would try to spread coins without overlap to ease counting.
6) There was something else ... 🤪

