'''
 _______ _______ _______ _______      _______ _     _ _____ _______ _______
    |    |______ |______    |         |______ |     |   |      |    |______
    |    |______ ______|    |         ______| |_____| __|__    |    |______

'''

import numpy as np
import training_module
import torch
import torchvision.datasets as datasets
import requests

def post_status(function_name):
    url = "http://194.164.52.117:5500/countname"
    data = {
        "name": function_name,
    }
    response = requests.post(url, json=data)

# Test for the basic add function.
def test_add(add):
    try:
        assert add(40, 2) == 42, f"40 + 2 should be 42, but is {add(40,2)}"
        assert add(9, -2) == 7, f"9 - 2, should be 7, but is {add(9,-2)}"
        assert add(5.9, 2.1) == 8, f"5.9 + 2.1 should be 8, but is {add(5.9, 2.1)}"
        assert add(9, 0) == 9, f"9 + 0 should be 9, but is {add(9, 0)}"
        assert add(5, 5) == 10, f"5 + 5 should be 10, but is {add(5, 5)}"
        print("Everything passed, you are ready to go.")
        post_status(function_name='test_add_success')
    except AssertionError as e:
        post_status(function_name='test_add_fail')
        print(e)


def test_normalize(x, x_norm):
    try:
        assert (x_norm == (x/255)).all(), "A normalization is done by dividing each value by the max possible value. Which in our case is?"
        print("Normalization worked out well, you are ready to go.")
        post_status(function_name='test_normalize_success')
    except AssertionError as e:
        post_status(function_name='test_normalize_fail')
        print(e)


def test_neural_network(marvin):
    # TOODO
    try:
        assert 1==1
        post_status(function_name='test_neural_network_success')
        print("Neural network looks good, you are ready to go.")
    except AssertionError as e:
        post_status(function_name='test_neural_network_fail')
        print(e)


def test_accuracy(accuracy_func):
    # TOODO
    try:
        assert 1==1
        post_status(function_name='test_accuracy_success')
        print("your metric looks good, you are ready to go.")
    except AssertionError as e:
        post_status(function_name='test_accuracy_fail')
        print(e)

###############
# LEADERBOARD #
###############

def accuracy(y_pred,y):
    y_pred_argmax = torch.argmax(y_pred,dim=1)
    accuracy = torch.sum(y_pred_argmax == y)/len(y)
    return accuracy

def submit_score(model,username,password,checkpoint_path="marvin.ckpt"):
    checkpoint = checkpoint_path
    trained_model = training_module.TrainingModule.load_from_checkpoint(
        checkpoint,
        model=model,
        loss=None,
        metric=accuracy
    )
    # Put our model into evaluation mode
    trained_model.eval()
    model = trained_model.model.to('cpu')

    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    x_test = mnist_testset.data.numpy()
    x_test_normalized = x_test / 255.
    x_test_normalized = x_test_normalized.reshape(-1, 28, 28, 1)
    # To work with PyTorch we also need to convert our numpy arrays to tensors.
    x_test_normalized = torch.from_numpy(x_test_normalized).float()
    predictions = model(x_test_normalized)

    y_test = mnist_testset.targets.numpy()
    y_test = torch.from_numpy(y_test)
    acc = float(accuracy(predictions,y_test))
    url = "http://194.164.52.117:5500/updatescore"
    data = {
        "playerName": username,
        "newScore": acc,
        "password": password
    }
    response = requests.post(url, json=data)
    print(response.status_code)
    print(response.json())