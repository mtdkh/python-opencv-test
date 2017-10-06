# -*- coding: utf-8 -*-

import cv2
import numpy as np
import glob
import pylab as plt




# オリジナル画像表示(BGR)
def org(img_org,img_name):
	cv2.imwrite(output_path+'ORG_'+img_name, img_org)
	cv2.imshow('img_org', img_org)
	cv2.waitKey(0)


# HSV変換
def hsv(img_org,img_name):
	img_hsv = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)
	cv2.imwrite(output_path+'HSV_'+img_name, img_hsv)
	cv2.imshow('img_hsv', img_hsv)
	cv2.waitKey(0)


# グレイスケール変換
def gry(img_org,img_name):
	img_gry = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
	cv2.imwrite(output_path+'GRY_'+img_name, img_gry)
	cv2.imshow('img_gry', img_gry)
	cv2.waitKey(0)


# グレイスケール画像でエッジ検出(輝度の強度50~150)
def edge(img_org,img_name):
	img_gry = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
	img_edge = cv2.Canny(img_gry,50,150)
	cv2.imwrite(output_path+'EDGE_'+img_name, img_edge)
	cv2.imshow('img_edge', img_edge)
	cv2.waitKey(0)


# ガウシアン平滑化(フィルタ範囲(15,15),度合い10)
def gaus(img_org,img_name):
	img_gaus = cv2.GaussianBlur(img_org,(15,15),10)
	cv2.imwrite(output_path+'GAUS_'+img_name, img_gaus)
	cv2.imshow('img_gaus', img_gaus)
	cv2.waitKey(0)


# 単純平滑化(フィルタ範囲(15,15))
def blur(img_org,img_name):
	img_blur = cv2.blur(img_org,(15,15))
	cv2.imwrite(output_path+'BLUR_'+img_name, img_blur)
	cv2.imshow('img_blur', img_blur)
	cv2.waitKey(0)


# 中央値平滑化(フィルタ範囲15)
def median(img_org,img_name):
	img_median = cv2.medianBlur(img_org,15)
	cv2.imwrite(output_path+'MEDIAN_'+img_name, img_median)
	cv2.imshow('img_median', img_median)
	cv2.waitKey(0)


# カラーヒストグラム(グレイスケール)
def gryhist(img_org,img_name):
	img_gry = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
	img_histgram = np.zeros([100, 256]).astype("uint8")
	rows, cols = img_histgram.shape
	histgram = cv2.calcHist([img_gry], [0], None, [256], [0,256])
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(histgram)

	for i in range(0, 255):
		v = histgram[i]
		cv2.line(img_histgram,
			(i, rows),
			(i, rows - rows * (v / max_val)),
			(255, 255, 255))

	cv2.imwrite(output_path+'GRYHIST_'+img_name, img_histgram)
	cv2.imshow("img_histgram", img_histgram)
	cv2.waitKey(0)


# 均一化処理とカラーヒストグラム
def eqhist(img_org,img_name):
	img_gry = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
	img_eqh = cv2.equalizeHist(img_gry)
	cv2.imwrite(output_path+'EQH_'+img_name, img_eqh)
	cv2.imshow('img_eqh', img_eqh)
	cv2.waitKey(0)

	img_eqhist = np.zeros([100, 256]).astype("uint8")
	rows, cols = img_eqhist.shape
	histgram = cv2.calcHist([img_eqh], [0], None, [256], [0,256])
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(histgram)

	for i in range(0, 255):
		v = histgram[i]
		cv2.line(img_eqhist,
			(i, rows),
			(i, rows - rows * (v / max_val)),
			(255, 255, 255))

	cv2.imwrite(output_path+'EQHIST_'+img_name, img_eqhist)
	cv2.imshow("img_eqhist", img_eqhist)
	cv2.waitKey(0)


# SIFT特徴点
def sift(img_org,img_name):
	img_gry = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create(nfeatures = 0, nOctaveLayers = 3, contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6)
	keypoints, descriptors = sift.detectAndCompute( img_gry, None )
	img_sift = np.empty_like(img_org)

	# cv2.drawKeypoints(img_org, keypoints, img_sift, -1, flags = cv2.DRAW_MATCHES_FLAGS_DEFAULT)
	cv2.drawKeypoints(img_org, keypoints, img_sift, -1, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	print(img_name+' = '+str(len(keypoints))+' keypoints')
	cv2.imwrite(output_path+'SIFT_'+img_name, img_sift)
	cv2.imshow('img_sift', img_sift)
	cv2.waitKey(0)


# pylabヒストグラム
def pylabhist(img_org,img_name):

	# 青緑赤のカラーヒストグラム	
	fig = plt.figure()
	fig.add_subplot(311)
	plt.xlim([0,256])
	plt.hist(img_org[:,:,0].ravel(), 256, range=(0, 255), fc='b')
	fig.add_subplot(312)
	plt.xlim([0,256])
	plt.hist(img_org[:,:,1].ravel(), 256, range=(0, 255), fc='g')
	fig.add_subplot(313)
	plt.xlim([0,256])
	plt.hist(img_org[:,:,2].ravel(), 256, range=(0, 255), fc='r')
	plt.savefig(output_path+'PYLABHIST_'+img_name)
	plt.show()


# HSVヒストグラム
def hsvhist(img_org,img_name):
	img_hsv = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)
	fig = plt.figure()
	fig.add_subplot(311)
	plt.xlim([0,256])
	plt.hist(img_hsv[:,:,0].ravel(), 256, range=(0, 160))
	fig.add_subplot(312)
	plt.xlim([0,256])
	plt.hist(img_hsv[:,:,1].ravel(), 256, range=(0, 255))
	fig.add_subplot(313)
	plt.xlim([0,256])
	plt.hist(img_hsv[:,:,2].ravel(), 256, range=(0, 255))
	plt.savefig(output_path+'HSVHIST_'+img_name)
	plt.show()




if __name__ == "__main__":

	# パス設定
	input_path = './InputImage/*.png'
	output_path = './OutputImage/'

	# ファイル名の取得
	img_src = glob.glob(input_path)

	# 画像データとファイル名の配列
	img_org = []
	img_name = []

	for i in range(0,len(img_src)):
		
		# 画像をOpenCVで扱える形に変換して配列に追加
		img_org.append(cv2.imread(img_src[i]))

		# ファイル名を配列に追加
		img_name.append(img_src[i][13:])

		# 画像処理(処理したい関数のコメントアウトを外してください)
		org(img_org[i],img_name[i])
		# hsv(img_org[i],img_name[i])
		# gry(img_org[i],img_name[i])
		# edge(img_org[i],img_name[i])
		# gaus(img_org[i],img_name[i])
		# blur(img_org[i],img_name[i])
		# median(img_org[i],img_name[i])
		# gryhist(img_org[i],img_name[i])
		# eqhist(img_org[i],img_name[i])
		# sift(img_org[i],img_name[i])
		# pylabhist(img_org[i],img_name[i])
		# hsvhist(img_org[i],img_name[i])