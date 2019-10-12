#include<bits/stdc++.h>
using namespace std;
char s[1000];
map<string,bool> ma;
int main(){
    FILE *fp=fopen("word=pre1.txt","r");
    FILE *fo=fopen("word1.txt","w");
    char t;
    int cnt=0;
    fscanf(fp,"%*c%*c%*c");
    while(fgets(s,1000,fp)){
        if(!ma[s]){
            ma[s]=true;
            if(cnt%5==0)fprintf(fo,"Find(");
            s[strlen(s)-1]=0;
            for(int i=0;s[i];++i)if(s[i]==95)s[i]=32;
            fprintf(fo,"\"%s\"",s);
            if(cnt%5==4)fprintf(fo,")\n");
            else fprintf(fo,",");
            if(cnt%20==19)fprintf(fo,"Random, x, 5000, 10000\nSleep %%x%%\n");
            ++cnt;
        }
    }
    while(cnt%5){
        fprintf(fo,"\"a\"");
        if(cnt%5==4)fprintf(fo,")\n");
        else fprintf(fo,",");
        ++cnt;
    }
    fclose(fo);
    fclose(fp);
    printf("%d\n",ma.size());
}
