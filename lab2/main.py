import os
import sys
def main():
    print("LSTM")
    os.system("python train.py --model LSTM")
    print("CNN")
    os.system("python train.py --model CNN")
    print('MLP')
    os.system("python train.py --model MLP")
    
if __name__ == "__main__":
    main()
    
    