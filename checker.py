'''
 _______ _______ _______ _______      _______ _     _ _____ _______ _______
    |    |______ |______    |         |______ |     |   |      |    |______
    |    |______ ______|    |         ______| |_____| __|__    |    |______

'''

import numpy as np

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


def test_forwardPass(forwardPass):
    assert forwardPass(np.array([[0.2], [4]]), np.array([[0.1, 0.5]])) == 4.0804, "Given x and W  the output signal should be 4.0804"
    print("Forward Pass was successful, you are ready to go.")


def test_objectiveFunction(objectiveFunction):
    assert objectiveFunction(0, 1) == 1, "Given prediction 0 and label 1 the loss should be 1"
    assert objectiveFunction(0, 0) == 0, "Given prediction 0 and label 0 the loss should be 0"
    assert objectiveFunction(0.5, 1) == 0.25, "Given prediction 0.5 and label 1 the loss should be 0.25"
    assert objectiveFunction(0.31415926, 1) == 0.47037752064374755, "Given prediction 0.31415926 and label 1 the loss should be 0.47037752064374755"
    print("Your Objective Function was successful, you are ready to go.")

def test_gradientFunction(gradientFunction):
    assert gradientFunction( np.array([[2.0]]), np.array([[0.2], [4]]), 3, 2).all() == np.array([[1.6, 32.]]).all(), "You broke it, check the dimensions."
    print("Gradient Function was successful, you are ready to go.")

def test_update(update):
    assert update(np.array([[ 1.6, 32. ]]), np.array([[ 1., 3. ]]), 1e-5).all() == np.array([[ 1.59999, 31.99997]]).all(), "Not quite right, the convention is to substract from the current weight."
    print("Update function works fine, you are ready to go.")

def test_normalize(x, x_norm):
    assert (x_norm == (x/255)).all(), "A normalization is done by dividing each value by the max possible value. Which in our case is?"
    print("Normalization worked out well, you are ready to go.")


import training_module
import torch
import torchvision.datasets as datasets
import requests

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