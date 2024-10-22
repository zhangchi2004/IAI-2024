#include "MCST.h"
#include<iostream>
bool userWin(const int x, const int y, const int M, const int N, int *const *board)
{
    //横向检测
    int i, j;
    int count = 0;
    for (i = y; i >= 0; i--)
        if (!(board[x][i] == 1))
            break;
    count += (y - i);
    for (i = y; i < N; i++)
        if (!(board[x][i] == 1))
            break;
    count += (i - y - 1);
    if (count >= 4)
        return true;

    //纵向检测
    count = 0;
    for (i = x; i < M; i++)
        if (!(board[i][y] == 1))
            break;
    count += (i - x);
    if (count >= 4)
        return true;

    //左下-右上
    count = 0;
    for (i = x, j = y; i < M && j >= 0; i++, j--)
        if (!(board[i][j] == 1))
            break;
    count += (y - j);
    for (i = x, j = y; i >= 0 && j < N; i--, j++)
        if (!(board[i][j] == 1))
            break;
    count += (j - y - 1);
    if (count >= 4)
        return true;

    //左上-右下
    count = 0;
    for (i = x, j = y; i >= 0 && j >= 0; i--, j--)
        if (!(board[i][j] == 1))
            break;
    count += (y - j);
    for (i = x, j = y; i < M && j < N; i++, j++)
        if (!(board[i][j] == 1))
            break;
    count += (j - y - 1);
    if (count >= 4)
        return true;

    return false;
}

bool machineWin(const int x, const int y, const int M, const int N, int *const *board)
{
    //横向检测
    int i, j;
    int count = 0;
    for (i = y; i >= 0; i--)
        if (!(board[x][i] == 2))
            break;
    count += (y - i);
    for (i = y; i < N; i++)
        if (!(board[x][i] == 2))
            break;
    count += (i - y - 1);
    if (count >= 4)
        return true;

    //纵向检测
    count = 0;
    for (i = x; i < M; i++)
        if (!(board[i][y] == 2))
            break;
    count += (i - x);
    if (count >= 4)
        return true;

    //左下-右上
    count = 0;
    for (i = x, j = y; i < M && j >= 0; i++, j--)
        if (!(board[i][j] == 2))
            break;
    count += (y - j);
    for (i = x, j = y; i >= 0 && j < N; i--, j++)
        if (!(board[i][j] == 2))
            break;
    count += (j - y - 1);
    if (count >= 4)
        return true;

    //左上-右下
    count = 0;
    for (i = x, j = y; i >= 0 && j >= 0; i--, j--)
        if (!(board[i][j] == 2))
            break;
    count += (y - j);
    for (i = x, j = y; i < M && j < N; i++, j++)
        if (!(board[i][j] == 2))
            break;
    count += (j - y - 1);
    if (count >= 4)
        return true;

    return false;
}

bool isTie(const int N, const int *top)
{
    bool tie = true;
    for (int i = 0; i < N; i++)
    {
        if (top[i] > 0)
        {
            tie = false;
            break;
        }
    }
    return tie;
}
using namespace std;
int main(){
    int M=3;
    int N=3;
    int** board = new int*[M];
    for(int i=0;i<M;i++){
        board[i] = new int[N];
        for(int j=0;j<N;j++){
            board[i][j]=0;
        }
    }
    int* top = new int[N];
    for(int i=0;i<N;i++){
        top[i] = M;
    }
    int noX = 1;
    int noY = 1;
    int lastX = -1;
    int lastY = -1;
    int x = 0;
    int y = 0;
    MCST* mcst = new MCST{board, top, M, N, noX, noY,-1,-1};
	Node* next = mcst->uctSearch();
	x = next->X;
	y = next->Y;
	delete mcst;
}