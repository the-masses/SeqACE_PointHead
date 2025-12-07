import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def superpoint_loss(scores, scores_label):
	# print(scores_label.shape)
	scores_label = scores_label.view(-1, scores_label.size(2), scores_label.size(3), scores_label.size(4))
	print(scores_label.shape)
	print(scores.shape)
	scores_label = scores_label.argmax(dim=1)
	print(scores_label.shape)
	loss_function = nn.CrossEntropyLoss()
	# loss_function = nn.NLLLoss()
	scores_loss = loss_function(scores, scores_label)

	return scores_loss