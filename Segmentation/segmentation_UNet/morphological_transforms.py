{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "\n",
    "def morph_trans(img):\n",
    "    kernel = np.array([50,50,50,50,50,50,50,50, 50], dtype=np.uint8)\n",
    "    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)\n",
    "    kernel = np.ones([15,15], dtype=np.uint8)\n",
    "    erosion = cv2.erode(closing, kernel, cv2.BORDER_REFLECT) \n",
    "    kernel = np.ones((20, 20), 'uint8')\n",
    "    dilate_img = cv2.dilate(erosion, kernel, iterations=1)\n",
    "    \n",
    "    return dilate_img"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
