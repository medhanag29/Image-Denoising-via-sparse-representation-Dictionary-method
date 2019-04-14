clear; close all;
 X=imread('ct_1.jpg');
 X=im2double(X);
  X= imresize(X,[256 256]);
  figure
  imshow(X)
  save('X');
