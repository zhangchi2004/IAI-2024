#pragma once

#include<iostream>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include<stdlib.h>
#include<chrono>
#include "Judge.h"
#include "Point.h"
// #define SHOW
using namespace std;

void show(int**board, int M, int N){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            if(board[i][j]==0)cout<<". ";
            else cout<<board[i][j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;

}

int find_x(int y, const int* top, int ban_x, int ban_y){
    int x = top[y]-1;
    if(x==ban_x&&y==ban_y){
        x--;
    }
    return x;
}

class Node{
public:
    int board[12][12];
    int top[12];
    int* board_ptrs[12];
    
    int M;
    int N;
    int ban_x;
    int ban_y;
    Node* parent = nullptr;
    Node* children[12];
    bool player = false; // true for the AI, false for the opponent
    bool tried[12];
    double wins = 0;
    double all = 0; // win_rate = wins/all
    int X = -1; // the move that leads to this node
    int Y = -1;
    bool leaf = true;
    Node()=default;

    Node(int** _board, const int* _top, const int _M, const int _N, const int _ban_x, const int _ban_y, Node* _parent=nullptr, bool _player=false, const int _X=-1, const int _Y=-1):M(_M), N(_N), ban_x(_ban_x), ban_y(_ban_y), parent(_parent), player(_player), X(_X), Y(_Y){

        for(int i=0;i<M;i++){
            for(int j=0;j<N;j++){
                board[i][j] = _board[i][j];
            }
        }
        for(int i=0;i<M;i++){
            board_ptrs[i] = board[i];
        }
        
        for(int i=0;i<N;i++){
            top[i] = _top[i];
        }
        
        for(int i=0;i<N;i++){
            tried[i] = 0;
        }
        
        for(int i=0;i<N;i++){
            children[i] = nullptr;
        }
        
        if(X!=-1){
            board[X][Y] = player?2:1;
            top[Y] = X;
        }
    }
    ~Node(){
        
        for(int i=0;i<N;i++){
            if(children[i])delete children[i];
        }
    }
    
    
    

    Node* expand(){
        int i=0;
        int new_y = -1;
        int new_x = -1;
        for(;i<N;i++){
            if(tried[i]==false){
                new_y = i;
                new_x = find_x(new_y,top,ban_x,ban_y);
                if (new_x>=0) break;
            }
        }
        if(i==N) {
            return nullptr;
        }
        tried[i] = true;
        
        children[i] = new Node(board_ptrs,top,M,N,ban_x,ban_y,this,!player,new_x,new_y);
        return children[i];
    }
    
    pair<Node*,Node*> best2Child(double C){
        double best_val = -1;
        double sec_val = -1;
        int best_idx = -1;
        int sec_idx = -1;
        for(int i=0;i<N;i++){
            if(!children[i])continue;
            double val = double(children[i]->wins)/double(children[i]->all) + C * sqrt(log(all) / double(children[i]->all));
            if(val > best_val){
                sec_idx = best_idx;
                sec_val = best_val;
                best_val = val;
                best_idx = i;
            }
            else if(val > sec_val){
                sec_val = val;
                sec_idx = i;
            }
        }
        if(sec_idx<0)sec_idx=1;
        if(best_idx<0)best_idx=0;
        return pair<Node*,Node*>(children[best_idx],children[sec_idx]);
    }
    


};

int find_x_MCST(int y, int* top, int ban_x, int ban_y){
    int x = --top[y];
    if(x==ban_x&&y==ban_y){
        x = --top[y];
    }
    if(top[y]<=0)top[y]=0;
    return x;
}

class MCST{
public:
    int M;
    int N;
    double C = 0.7;
    int ban_x;
    int ban_y;
    Node* root;

    MCST(int** _board, const int* _top, const int _M, const int _N, const int _ban_x, const int _ban_y, const int _X=-1, const int _Y=-1):M(_M), N(_N), ban_x(_ban_x), ban_y(_ban_y){
        root = new Node(
            _board, _top, _M, _N, _ban_x, _ban_y, nullptr, false, _X, _Y
        );
    }

    ~MCST(){
        delete root;
    }

    Node* findLeaf(){
        Node* v = root;
        while(true){
            Node* v1 = v->expand();
            if(v1)return v1;
            else v = v->best2Child(C).first;
            if(v->player&&machineWin(v->X,v->Y,M,N,v->board_ptrs))return v;
            if(!v->player&&userWin(v->X,v->Y,M,N,v->board_ptrs))return v;
        }
        return nullptr;
    }

    

    int simulate(Node* v){
        int board[M][N];
        int* board_ptrs[M];
        for(int i=0;i<M;i++){
            board_ptrs[i] = board[i];
        }
        for(int i=0;i<M;i++){
            for(int j=0;j<N;j++){
                board[i][j] = v->board[i][j];
            }
        }
        int top[N];
        for(int i=0;i<N;i++){
            top[i] = v->top[i];
        }
        
        bool player = v->player; // the player just done, true for AI
        int r = 0;
        int rx = v->X;
        int ry = v->Y;
        while(true){
            if(userWin(rx,ry,M,N,board_ptrs)){r=-1;break;}
            if(machineWin(rx,ry,M,N,board_ptrs)){r=1;break;}
            if(isTie(N,top)){r=0;break;}
            ry = rand()%N;
            rx = find_x_MCST(ry,top,ban_x,ban_y);
            int c=0;
            while(rx<0){
                ry++;
                c++;
                if(c>=N){
                    // for(int i=0;i<N;i++){cout<<top[i]<<" ";}
                    break;}
                if(ry==N)ry=0;
                rx = find_x_MCST(ry,top, ban_x,ban_y);
            }

            player = !player;
            board[rx][ry] = player?2:1;
        }
        return r;
    }

    void backward(Node* v, int reward){
        while(v){
            v->all++;
            if((v->player)&&(reward==1))v->wins++;
            else if ((!v->player)&&(reward==-1))v->wins++;
            // only calculate wins, no ties counted

            // v->all+=10;
            // int r = reward;
            // r = (r+10)/2;
            // if(v->player){
            //     v->wins+=r;
            // }
            // else{
            //     v->wins+=10-r;
            // }
            v = v->parent;
        }
    }
    Node* uctSearch(){
        auto start_time = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
        int i=0;
        
        while(true){
            i++;
            auto cur_time = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
            if(cur_time - start_time > 2500){
                // cout<<"use time "<<cur_time - start_time<<endl;
                break;
            }
            Node* v = findLeaf();
            // int r =0;
            // for(int j=0;j<10;j++){
            //     r += simulate(v);
            // }
            int r = simulate(v);
            backward(v,r);
        }
        auto b2c = root->best2Child(0);
        auto bc = b2c.first;
        auto r = bc;
        #ifdef SHOW
        cout<<b2c.first->X<<" "<<b2c.first->Y<<endl;
        if(b2c.second)cout<<b2c.second->X<<" "<<b2c.second->Y<<endl;
        #endif

        // int iy = bc->Y;
        // int ix = find_x(bc->Y, bc->top, ban_x, ban_y);
        // if(ix>=0){
        //     bc->board[ix][iy] = 1;
        //     if(userWin(ix,iy,M,N,bc->board_ptrs)){
        //         if(b2c.second)r = b2c.second;
        //         #ifdef SHOW
        //         cout<<endl<<"defending!!!"<<endl<<endl;
        //         #endif
        //     }
        //     bc->board[ix][iy] = 0;
        // }

        #ifdef SHOW
        cout<<"search times "<<i<<endl;
        
        // for (int i=0;i<N;i++){
        //     if(root->children[i]){
        //         cout<<root->children[i]->wins<<" "<<root->children[i]->all<<endl;
        //     }
        // // }
        
        cout<<r->X<<" ";
        cout<<r->Y<<" to act"<<endl;
        cout<<"The board before:"<<endl;
        show(root->board_ptrs,M,N);
        cout<<"The board after:"<<endl;
        show(r->board_ptrs,M,N);
        #endif
        
        return r;
    }
};

