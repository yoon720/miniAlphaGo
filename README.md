# mini AlphaGo Zero
* Tensorflow implementation of mini AlphaGo Zero
* Final project of "EE807 Deep Reinforcement Learning & AlphaGo", 2019 Fall, School of EE, KAIST
* Improved version of https://github.com/s-chung/mini-alphago
  - increased the board size to 6x6
  - modified the neural network such that it outputs a policy


## 1. Class & functions
1) `class game1_new`: added new function `move` to `game1` from `boardgame.py` such that gets position k and puts stone on it.
2) `class Node`: a class for MCTS. Each node has value U, N, V; child; and functions for tree search
3) `data_augmentation` : get current board and make 7 variations such that original, horizontal flip, vertical flip, both, 90/180/270 degree rotations
4) `train` : initialize the neural network, generate a training dataset, train the current neural network, and repeat step 2, 3. Then save `*.ckpt` files
5) `test_random` : load `*.ckpt` and play 50,000 games as black and 50,000 games as white against a pure random policy
6) `test_two` : load two `*.ckpt` files and play a game between them, where the first `*.ckpt` is black and the second is white
  
  
## 2. Hyperparameter
한 mini batch에서 training data를 적게 사용하기 때문에 learning rate를 0.0005로 조금 낮춰주었다. 각 node에 대해 explore는 충분히 돌게 하기 위해 default np0인 nx\*ny\*2보다 큰 100으로 정했다. Max epoch는 충분히 크게 하기 위해 500으로 잡았다.


## 3. Neural net
![figure1](https://user-images.githubusercontent.com/52485688/87569492-2b632000-c702-11ea-9a0e-5cd96d92416e.png)
(reference: lecture note 11, 46p)

Network 구조는 위 그림을 참고하였다.  Residual block 대신 간단하게 convolutional layer 3개를 사용하였고 그림과 같이 한 network로부터 policy, value를 얻을 수 있도록 구성하였다. Policy 부분의 첫 번째 fully connected layer에 student id의 마지막 숫자를 사용하였다.

## 4. Training method
![figure2](https://user-images.githubusercontent.com/52485688/87569498-2d2ce380-c702-11ea-84d3-2172d8c7b6d3.PNG)

(reference: lecture note 11, 53 p)

train 함수에서 training이 진행된다. Training 방법은 위의 그림을 참고하였다.  각 epoch마다 새로운 tree를 만들고 search를 시작한다. 100번의 search 과정을 거친 후 next 함수를 사용하면 가장 확률이 높은 다음 state를 가진 node와 training data로 사용할 d, v, p (그림의 s, z, pi)를 얻을 수 있다. 얻은 d, v, p를 8배로 augmentation 해서 training data를 늘리고 network를 훈련하는데 이를 하나의 mini batch로 볼 수 있다. 이러한 과정을 tree가 terminal state에 도달할 때까지 진행하면 하나의 epoch가 진행된 것이고 이를 500회 반복하도록 구성되어 있다.

https://github.com/s-chung/mini-alphago 로부터 `boardgame.py`를 받은 후 다음과 같이 입력하여 model train 시작할 수 있다. train 과정에서 10 epoch마다 `./ckpt` 폴더에 `project2_task3_20193404.ckpt`를 갱신한다.
```bash
python project2_task_20193404_train.py
```

## 5. Test result against random
![figure3](https://user-images.githubusercontent.com/52485688/88543190-8ece4580-d052-11ea-9d45-b90be7929e81.png)

구현한 test_random 함수를 이용하여 black, white로 각 50000번씩 대결했을 때 각각 98.1%, 95.0%의 승률을 보였다.



### Reference:
https://github.com/s-chung/mini-alphago  
https://jsideas.net/AlphaZero/
