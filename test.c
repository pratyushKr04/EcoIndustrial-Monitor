#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

int main(){

    int *arr=NULL;
    int curr_size=0;
    void add_element(int n){
        arr=realloc(arr,curr_size+1);
        arr[curr_size]=n;
        curr_size++;
    }
    void iterate(){
        for(int i=0;i<curr_size;i++){
            printf("%d\n",arr[i]);
        }
    }

    add_element(1);
    add_element(2);
    add_element(10);
    add_element(4);
    add_element(5);
    add_element(6);
    add_element(7);
    add_element(8);
    printf("size - %d \n",curr_size);
    iterate();

    return 0;
}