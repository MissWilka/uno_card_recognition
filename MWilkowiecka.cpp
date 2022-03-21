#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;


float w = 300, h = 400;	//szerokoœæ i wysokoœæ przeskalowanej karty
Point2f dst[4] = { {0.0f,0.0f}, {w,0.0f},{w,h},{0.0f,h} };	//punkty docelowe przeskalowanej karty


bool blurDetector(Mat& imgGray, Mat& lapl) //zwraca wartoœc true jeœli obraz jest rozmazany, false jeœli nie jest
{
	Laplacian(imgGray, lapl, CV_32FC1);
	Scalar mean, stddev;
	meanStdDev(lapl, mean, stddev, imgGray);
	double variance = stddev.val[0] * stddev.val[0];
	double threshold = 10;
	if (variance <= threshold) return true;
	else return false;
}

void PreProcessing(Mat img, Mat& imgCanny, Mat& imgGray)
{
	Mat tmp,lapl;	//Macierze tymczasowe, do wykonywania operacji w funckji
	cvtColor(img, imgGray, COLOR_BGR2GRAY);	//Konwertowanie na skalê szaroœci
	blurDetector(imgGray, lapl);	//wykrywanie rozmazania
	double alpha = 4;	//parametr alpha
	int threshUp = 200;	//górna wartoœæ thresholdu
	int threshDown = 100;	//dolna wartoœæ threshldu

	if (blurDetector(imgGray, lapl))
	{
		for (int i = 0; i < 3; i++)
		{
		//zastosowanie filtru unsharpmask dla najlepszych rezultatów - trzykrotne
		GaussianBlur(imgGray, tmp, Size(301, 301), 2.0, 2.0);
		addWeighted(imgGray, 1 + alpha, tmp, -alpha, 0, imgGray);
		}		
		threshDown = 140; // nowa dolna wartoœæ thresholu - jeœli obraz rozmazany
		threshUp = 255;	//nowa górna wartoœc thresholdu - jeœli obraz rozmazany
		medianBlur(imgGray, imgGray, 5);	//usuniêcie mo¿liwego, nowopowsta³ego szumu
	}
	else
	{
		medianBlur(imgGray, imgGray, 7); //usuniêcie szumu
	}

	Canny(imgGray, imgCanny, threshDown, threshUp);	//stworzenie obrazu imgCanny

}

void ImgCountours(Mat& img, Mat& imgGray, Mat& imgCanny, vector<Mat>& imgWarp, vector<RotatedRect>& MinRect)
{
	vector<vector<Point>> contours;	//utworzenie stuktury konturów
	vector<Vec4i> hierarchy;	//utworzenie struktury hierarchii
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));	//stworzenie elementu strukturalnego do operacji dylatacji
	dilate(imgCanny, imgCanny, kernel);	//operacja dylatacji
	dilate(imgCanny, imgCanny, kernel);	//powtórzenie operacji dyatacji

	findContours(imgCanny, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); //znajdowanie konturów
	vector<RotatedRect> minRect(contours.size());	
	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());	//stworzenie elementów potrzebnych do dalszych operacji
	int area;	//wartoœæ obszaru ogarniczonego konturami
	Point2f src[4] = { {0.0f,0.0f}, {0.0f,0.0f},{0.0f,0.0f},{0.0f,0.0f} };	//wspó³rzêdne Ÿród³a
	
	
	for (int i = 0; i < contours.size(); i++)	//przejœcie przez wszystkie kontury
	{
		area = contourArea(contours[i]);	//przypisanie wartoœci obszaru ograniczonemu konturami

		if (area >= 300000)	//warunek rozmiaru obszaru
		{
			minRect[i] = minAreaRect(contours[i]);
			Point2f crop_points[4];
			minRect[i].points(crop_points);
			for (int j = 0; j < 4; j++)
			{
				line(img, crop_points[j], crop_points[(j + 1) % 4], Scalar(0, 125, 255), 15);	//rysowanie obwodu ³¹cz¹c punkty
			}
			
			//sprawdzenie i poprawa orientacji znalezionej karty (z poziomej na pionow¹
			if (norm(crop_points[0] - crop_points[1]) < norm(crop_points[0] - crop_points[3]))	
			{
				for (int j = 0; j < 4; j++)
				{
					src[j].x = crop_points[j].x;
					src[j].y = crop_points[j].y;
				}
			}
			else
			{
				for (int j = 0; j < 4; j++)
				{
					if (j == 0)
					{
						src[j].x = crop_points[j + 3].x;
						src[j].y = crop_points[j + 3].y;
					}
					else
					{
						src[j].x = crop_points[j - 1].x;
						src[j].y = crop_points[j - 1].y;
					}
				}
			}

			Mat matrix, Img_warp;
			matrix = getPerspectiveTransform(src, dst);
			warpPerspective(img, Img_warp, matrix, Point(w, h));	//przesuniêcie karty do pocz¹tku uk³adu wspó³rzêdnych
			imgWarp.push_back(Img_warp);	//push_back zapobiega b³êdowi braku miejsca w wektorze
		}
	}

	MinRect = minRect;

}

void maska_karty(Mat& karta)	//maska wycinaj¹ca nam symbol z karty do dalszej operacji
{

	Point mid = { 70,65 };	//œrodek okrêgu
	Mat mask, after;
	mask = Mat::zeros(h, w, CV_8UC3);	//stworzenie maski

	circle(mask, mid, 45, { 255,255,255 }, -1);	//stworzenie okrêgu
	bitwise_and(karta, mask, karta);	//zastosowanie maski

}

string Symbol(float& mom2, float& mom3, string& symbol)	//przypisywanie symbolu do momentów hu
{
	if (((mom2 <= 5.1) || (mom2 >= 7.35 && mom2 <= 7.6)) && ((mom3 >= 7 && mom3 <= 7.1) || (mom3 > 11.5 && mom3 < 14.2)))
	{
		symbol = "0";
	}
	if (((mom2 > 5.65 && mom2 < 5.8)||(mom2>5.9&&mom2<6.66)) && ((mom3 > 8.3 && mom3 < 8.4) || (mom3 > 8.6 && mom3 < 9.35) || (mom3 >9.36 && mom3 <9.7 )))
	{
		symbol = "1";
	}
	if (((mom2>5.8 && mom2<5.9)||(mom2 > 7.25 && mom2 < 7.74)) && ((mom3 > 8.5 && mom3 < 8.7) || (mom3 > 10.5 && mom3 < 11.2)))
	{
		symbol = "8";
	}
	if ((mom2 > 5.9 && mom2 < 6.36) && ((mom3 > 8.4 && mom3 < 8.6) || (mom3 > 11.6 && mom3 < 13.1)))
	{
		symbol = "Reverse";
	}
	if (symbol == "symbol")
	{
		symbol = "Stop";
	}
	return symbol;	//zwrot znalezionego symbolu
}

string momenty(Mat& pre, string& symbol)	//znajdywanie HuMoments dla wycinka z mask¹
{
	float mom2, mom3;	//zmienne przekazuj¹ce dalej HuMoment[1] i HuMoment[2]
	Mat post;	//obraz tymczasowy, do konwersji na skale szaroœci
	int med = 0;	//implementacja œredniej wartoœci nasycenia - w skali szaroœci
	//poniewa¿ bêdzie ona uzyta do funkcj threshold (niweluje b³¹d zwi¹zany z gradientem)
	//dlatego jest liczb¹c ca³kowit¹
	cvtColor(pre, post, COLOR_BGR2GRAY);	//transformacja kolorystyczna do sklai szaroœci - obraz pre do post
	for (int i = 0; i < post.rows; i++)
	{
		for (int j = 0; j < post.cols; j++)
		{
			med = med + post.at<uchar>(i, j);	//zliczanie sumarycznej wartoœci wszystkich pixeli w wycinku - w skali szaroœci
		}
	}
	med = med / (post.rows * post.cols);	//wyliczenie œredniej
	med = med + 40;	//zwiêkszenie œredniej o 40
	medianBlur(post, post, 7);	//filt medianowy medianowe	- niwelacja b³edu szumu sól z pieprzem
	threshold(post, post, med, 255, THRESH_BINARY);	//binaryzacja obrazu w zale¿noœci od œredniego nasycenia pixeli
	medianBlur(post, post, 5);	//filt medianowy, do usuniêcia pozosta³ych szumów i uzupe³nienia ubytków
	Moments momenty = moments(post, false);	//tworzenie momentów
	double huMomenty[7];	
	HuMoments(momenty, huMomenty);	//wyliczanie momentów
	for (int i = 0; i < 7; i++)
	{
		huMomenty[i] = -1 * copysign(1.0, huMomenty[i]) * log10(abs(huMomenty[i]));	//zapisywanie momentów
	}
	mom2 = huMomenty[1];//przypisanie wartoœci drugiego momentu do zmiennej mom2 przekazywanej dalej
	mom3 = huMomenty[2];//przypisanie wartoœci trzeciego momentu do zmiennej mom3 przekazywanej dalej
	Symbol(mom2, mom3, symbol);	//odnajdywanie symboli
	return symbol;	//zwrot sybolu
}

void PostCropping(Mat& img, Mat& pre, string& color, string& symbol) //odnajdywanie koloru karty
{
	double alpha = 4;	//wspó³czynnik alfa
	double ch1 = 0;		//wartoœæ intensywnoœci pierwszego kana³u (B)
	double ch2 = 0;		//wartoœæ intensywnoœci drugiego kana³u (G)
	double ch3 = 0;		//wartoœæ intensywnoœci trzeciego kana³u (R)
	Mat tmp;			//macierz tymczasowa do dalszych operacji
	int range = 150;	//wartoœæ któr¹ musi przekroœcyæ œrednia intensywnoœc kana³u, aby uznaæ, ¿e dany kolor wystêpuje
	if (blurDetector(img, tmp))		//wykrywanie rozmycia obrazu
	{
		for (int p = 0; p < 3; p++) //wyostrzanie wycietego fragmentu
		{
			GaussianBlur(pre, tmp, Size(3, 3), 1, 1);
			addWeighted(pre, 1 + alpha, tmp, -alpha, 0, pre);
		}
	}
	threshold(pre, tmp, 90, 255, THRESH_BINARY);	//binaryzacja (wszystkie kana³y jeden po drugim)
	for (int i = 0; i < tmp.rows; i++)
	{
		for (int j = 0; j < tmp.cols; j++)	//przejœcie po ca³ym wycinku i zliczenie sumarycznej wartoœæi intensywnoœci dla poszczególnych kana³ów
		{
			ch1 = ch1 + tmp.at<Vec3b>(i, j)[0];
			ch2 = ch2 + tmp.at<Vec3b>(i, j)[1];
			ch3 = ch3 + tmp.at<Vec3b>(i, j)[2];
		}
	}
	alpha = tmp.rows * tmp.cols;	//ponowne u¿ycie wspó³czynnika alfa aby nie tworzyæ niepotrzebnych zmiennych
	ch1 = ch1 / alpha;				//wartoœæ sredniej intensywnoœci pierwszego kana³u
	ch2 = ch2 / alpha;				//wartoœæ sredniej intensywnoœæi drugiego kana³u
	ch3 = ch3 / alpha;				//wartoœæ œredniej intensywnoœci trzeciego kana³u


	//warunkowe przypisanie koloru w zale¿noœci od œredniej intensywnoœci kana³ów
	if (ch1 >= range)			
	{
		color = "Blue";
	}
	else
	{
		if (ch2 >= range)
		{
			if (ch3 >= range)
			{
				color = "Yellow";
			}
			else
			{
				color = "Green";
			}
		}
		else
		{
			color = "Red";
		}
	}
	momenty(pre, symbol);	//wyliczenie HuMoments dla tych samych wycinków
}

void main()
{

	string path = "images/4.png";		//wczytanie scie¿ki obrazu
	Mat imgOriginal = imread(path);				//wczytanie obrazu
	vector<Mat> imgWarp;						//stworzenie wektora do przekszta³cania kart
	Mat imgThre, imgCanny, imgPost;				//stworzenie macierzy na inne obrazy
	vector<RotatedRect> minRect;				//stworzenie wektora minRec			
	string color1 = "color";					//inicjalizacja kolorów dla 4 kart - domyœlnie "color"
	string color2 = "color";
	string color3 = "color";
	string color4 = "color";
	string symbol1 = "symbol";					//inicjalizacja symboli dla 4 kart - domyœlnie "symbol"
	string symbol2 = "symbol";
	string symbol3 = "symbol";
	string symbol4 = "symbol";

	PreProcessing(imgOriginal, imgCanny, imgPost);	//PreProcessing, przygotowanie karty do znalezenia konturów
	ImgCountours(imgOriginal, imgPost, imgCanny, imgWarp, minRect);	//znajdowanie konturów

	Mat karta1 = imgWarp[0];		//tworzenie i wpisywanie macierzy dla znalezionych kart
	Mat karta2 = imgWarp[1];
	Mat karta3 = imgWarp[2];
	Mat karta4 = imgWarp[3];

	Mat k1 = karta1.clone();		//tworzenie klonów znalezionych kart do póŸniejszego wyœwietlenia
	Mat k2 = karta2.clone();
	Mat k3 = karta3.clone();
	Mat k4 = karta4.clone();

	maska_karty(karta1);			//tworzenie i nak³adanie maski na wszytskie karty
	maska_karty(karta2);
	maska_karty(karta3);
	maska_karty(karta4);

	Rect roi(30, 30, 80, 80);		//Region Of Intrest - obszar w którym, w orientacji pionowej znajduje siê symbol karty

	Mat imgCrop1 = imgWarp[0](roi);	//tworzenie wycinków kart do rozpoznania kolorów i symboli
	Mat imgCrop2 = imgWarp[1](roi);
	Mat imgCrop3 = imgWarp[2](roi);
	Mat imgCrop4 = imgWarp[3](roi);

	cvtColor(imgOriginal, imgPost, COLOR_BGR2GRAY);	//konwersja obrazu oryginalnego na skalê szaroœci

	PostCropping(imgPost, imgCrop1, color1, symbol1);		//przypisanie wycinkom kolorów i symboli
	PostCropping(imgPost, imgCrop2, color2, symbol2);
	PostCropping(imgPost, imgCrop3, color3, symbol3);
	PostCropping(imgPost, imgCrop4, color4, symbol4);

	imshow(color1 + " " + symbol1, k1);			//pokazanie rozpoznanych kart w oknach nazwanych zgodnie z rozpoznanym kolorem i symbolem 
	imshow(color2 + " " + symbol2, k2);
	imshow(color3 + " " + symbol3, k3);
	imshow(color4 + " " + symbol4, k4);

	waitKey(0);

}