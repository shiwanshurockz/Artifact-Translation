import os.path
import json
from flask import Flask, request, Response
import uuid
from Utils import *
from keras.layers import *
import numpy as np
from keras.models import Model
from keras.models import model_from_json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transliteration_EncoderDecoder_Attention(nn.Module):

	def __init__(self, input_size, hidden_size, output_size, verbose=False):
		super(Transliteration_EncoderDecoder_Attention, self).__init__()

		self.hidden_size = hidden_size
		self.output_size = output_size

		self.encoder_rnn_cell = nn.GRU(input_size, hidden_size)
		self.decoder_rnn_cell = nn.GRU(hidden_size * 2, hidden_size)

		self.h2o = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=2)

		self.U = nn.Linear(self.hidden_size, self.hidden_size)
		self.W = nn.Linear(self.hidden_size, self.hidden_size)
		self.attn = nn.Linear(self.hidden_size, 1)
		self.out2hidden = nn.Linear(self.output_size, self.hidden_size)

		self.verbose = verbose

	def forward(self, input, max_output_chars=30, device='cpu', ground_truth=None):

		# encoder
		encoder_outputs, hidden = self.encoder_rnn_cell(input)
		encoder_outputs = encoder_outputs.view(-1, self.hidden_size)

		if self.verbose:
			print('Encoder output', encoder_outputs.shape)

		# decoder
		decoder_state = hidden
		decoder_input = torch.zeros(1, 1, self.output_size).to(device)

		outputs = []
		U = self.U(encoder_outputs)

		if self.verbose:
			print('Decoder state', decoder_state.shape)
			print('Decoder intermediate input', decoder_input.shape)
			print('U * Encoder output', U.shape)

		for i in range(max_output_chars):

			W = self.W(decoder_state.view(1, -1).repeat(encoder_outputs.shape[0], 1))
			V = self.attn(torch.tanh(U + W))
			attn_weights = F.softmax(V.view(1, -1), dim=1)

			if self.verbose:
				print('W * Decoder state', W.shape)
				print('V', V.shape)
				print('Attn', attn_weights.shape)

			attn_applied = torch.bmm(attn_weights.unsqueeze(0),
									 encoder_outputs.unsqueeze(0))

			embedding = self.out2hidden(decoder_input)
			decoder_input = torch.cat((embedding[0], attn_applied[0]), 1).unsqueeze(0)

			if self.verbose:
				print('Attn LC', attn_applied.shape)
				print('Decoder input', decoder_input.shape)

			out, decoder_state = self.decoder_rnn_cell(decoder_input, decoder_state)

			if self.verbose:
				print('Decoder intermediate output', out.shape)

			out = self.h2o(decoder_state)
			out = self.softmax(out)
			outputs.append(out.view(1, -1))

			if self.verbose:
				print('Decoder output', out.shape)
				self.verbose = False

			max_idx = torch.argmax(out, 2, keepdim=True)
			if not ground_truth is None:
				max_idx = ground_truth[i].reshape(1, 1, 1)
			one_hot = torch.zeros(out.shape, device=device)
			one_hot.scatter_(2, max_idx, 1)

			decoder_input = one_hot.detach()

		return outputs

def load_model(strr):
	json_file = open(strr, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	return loaded_model
eng_alphabets = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
pad_char = '-PAD-'

eng_alpha2index = {pad_char: 0}
for index, alpha in enumerate(eng_alphabets):
	eng_alpha2index[alpha] = index + 1
# Hindi Unicode Hex Range is 2304:2432. Sourc https://en.wikipedia.org/wiki/Devanagari_(Unicode_block)

hindi_alphabets = [chr(alpha) for alpha in range(2304, 2432)]
hindi_alphabet_size = len(hindi_alphabets)
hindi_alpha2index = {pad_char: 0}
for index, alpha in enumerate(hindi_alphabets):
	hindi_alpha2index[alpha] = index + 1
def predict_func(model, inp, iou, rimg):
	ans = model.predict(inp)
	img_w, img_h = (512, 512)
	boxes = decode(ans[0], img_w, img_h, iou)
	img = ((inp + 1) / 2)
	img = img[0]
	texts = list()
	for i in boxes:
		i = [int(x) for x in i]
		texts.append([i[0], i[1], i[2], i[3]])
	return texts


# Transliteration
def test(net, word, device='cpu'):
	net = net.eval().to(device)
	outputs = infer(net, word, 30, device)
	eng_output = ''
	for out in outputs:
		val, indices = out.topk(1)
		index = indices.tolist()[0][0]
		if index == 0:
			break
		eng_char = eng_alphabets[index - 1]
		eng_output += eng_char
	print(word + ' - ' + eng_output)
	return eng_output


def word_rep(word, letter2index, device='cpu'):
	rep = torch.zeros(len(word) + 1, 1, len(letter2index)).to(device)
	for letter_index, letter in enumerate(word):
		pos = letter2index[letter]
		rep[letter_index][0][pos] = 1
	pad_pos = letter2index[pad_char]
	rep[letter_index + 1][0][pad_pos] = 1
	return rep

def gt_rep(word, letter2index, device='cpu'):
	gt_rep = torch.zeros([len(word) + 1, 1], dtype=torch.long).to(device)
	for letter_index, letter in enumerate(word):
		pos = letter2index[letter]
		gt_rep[letter_index][0] = pos
	gt_rep[letter_index + 1][0] = letter2index[pad_char]
	return gt_rep

def infer(net, input, max_length, device='cpu'):
	input_rep = word_rep(input, hindi_alpha2index).to(device)
	outputs = net(input=input_rep)
	return outputs


# TEXT Reading
hindi_alphabets = [chr(alpha) for alpha in range(2304, 2432)]
hind_char_list = str()
for j in hindi_alphabets:
	hind_char_list += str(j)

def encode_to_labels(txt):
	# encoding each output word into digits
	dig_lst = []
	for index, char in enumerate(txt):
		try:
			dig_lst.append(hind_char_list.index(char))
		except:
			print(char)
	return dig_lst


def faceDetect(img):
	# Instantiates the device to be used as GPU/CPU based on availability
	device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# input with shape of height=32 and width=128
	inputs = Input(shape=(32, 128, 1))
	# convolution layer with kernel size (3,3)
	conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
	# poolig layer with kernel size (2,2)
	pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
	conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
	pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
	conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)
	conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
	# poolig layer with kernel size (2,1)
	pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
	conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
	# Batch normalization layer
	batch_norm_5 = BatchNormalization()(conv_5)
	conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
	batch_norm_6 = BatchNormalization()(conv_6)
	pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
	conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)
	squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
	# bidirectional LSTM layers with units=128
	blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
	blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)
	outputs = Dense(len(hind_char_list) + 1, activation='softmax')(blstm_2)
	# model to be used at test time

	act_model = Model(inputs, outputs)
	# load the saved best model weights

	os.chdir('E:\\WORK\\mAJOR\\Deployment\\weights')
	act_model.load_weights('best_model.hdf5')
	model = load_model('E:\\WORK\\mAJOR\\Deployment\\weights\\model\\text_detect_model.json')
	model.load_weights('E:\\WORK\\mAJOR\\Deployment\\weights\\text_detect.h5')

	pkl_filename = "pickle_model.pkl"
	with open(pkl_filename, 'rb') as file:
		pickle_model = pickle.load(file)

	img_w, img_h = (512, 512)

	MAX_OUTPUT_CHARS = 30

	rimg = cv2.resize(img, (512, 512))
	oimg = (rimg - 127.5) / 127.5
	texts_boxes = predict_func(model, np.expand_dims(oimg, axis=0), 0.5, rimg)
	if len(texts_boxes) <= 0:
		print('no text segmented')
	else:
		print('total no of text segmented:', len(texts_boxes))
	# loop through detected faces
	frame = cv2.resize(img, (512, 512))
	for f in texts_boxes:
		(startX, startY) = f[0], f[1]
		(endX, endY) = f[2], f[3]
		# draw rectangle over face
		cv2.rectangle(frame, (f[0], f[1]), (f[2], f[3]), (0, 255, 0), 2)
		# crop the detected face region
		text_crop = frame[f[1]:f[3], f[0]:f[2]]

		if (text_crop.shape[0]) < 10 or (text_crop.shape[1]) < 10:
			continue
		# preprocessing for TEXT READING
		dim = (128, 32)
		img = cv2.cvtColor(text_crop, cv2.COLOR_BGR2GRAY)
		# convert each image of shape (32, 128, 1)
		w, h = img.shape
		img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
		img = np.expand_dims(img, axis=2)
		# Normalize each image
		t_valid_img = list()
		img = img / 255.
		t_valid_img.append(img)
		t_valid_img = np.array(t_valid_img)
		# apply TEXT detection
		# predict outputs on validation images
		prediction = act_model.predict(t_valid_img)
		# use CTC decoder
		out = K.get_value(
			K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1], greedy=True)[0][
				0])
		res = list()
		for x in out:
			temp = ''
			for p in x:
				if int(p) != -1:
					temp += str(hind_char_list[int(p)])
			res.append(temp)
			print('\n')
		# get label with max accuracy
		label = res[0]
		# TRANSLITERATION
		label = test(pickle_model, label)

		# label = "{}: {:.2f}%".format(label)
		Y = startY - 10 if startY - 10 > 10 else startY + 10
		# write label and confidence above face rectangle
		cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
		# save File
	os.chdir('E:\\WORK\\mAJOR\\Deployment')
	path_file = ('static/%s.jpg'%uuid.uuid4().hex)
	cv2.imwrite(path_file,frame)
	return json.dumps(path_file)
#API
app = Flask(__name__)

@app.route('/api/upload', methods = ['POST'])
def upload():
	img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
	#PROCESS IMAGE
	img_processed = faceDetect(img)
	#response
	return Response(response=img_processed, status=200, mimetype="application/json")
#start server
app.run(host = "0.0.0.0", port = 5000)