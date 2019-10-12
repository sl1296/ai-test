#include<bits/stdc++.h>
using namespace std;
char s[1000];
char t[1000];
char ka[1000]=" - Google Ñ§ÊõËÑË÷.html";
char kb[1000]=" - Google ËÑË÷.html";
map<string,bool> ma;
typedef long long ll;
char ex[1000];
char tmp[10000000];
int cc;
ll getgg(){
    strcpy(t,"ggxs1\\");
    strcpy(&t[strlen(t)],s);
    strcpy(&t[strlen(t)],ka);
    FILE *fp=fopen(t,"r");
    if(!fp)return -1LL;
    ll ret=0;
    cc=0;
    while(~fscanf(fp,"%c",&tmp[cc]))++cc;
    fclose(fp);
    for(int i=0;i<cc;++i){
        bool ch=true;
        for(int j=0;ex[j];++j){
            if(tmp[i+j]!=ex[j]){
                ch=false;
                break;
            }
        }
        if(ch){
            i+=strlen(ex);
            while(tmp[i]==' '||tmp[i]==','||(tmp[i]>='0'&&tmp[i]<='9')){
                if(tmp[i]>='0'&&tmp[i]<='9'){
                    ret=ret*10+tmp[i]-48;
                }
                ++i;
            }
            return ret;
        }
    }
    return -2LL;
}
ll getggxs(){
    strcpy(t,"gg1\\");
    strcpy(&t[strlen(t)],s);
    strcpy(&t[strlen(t)],kb);
    FILE *fp=fopen(t,"r");
    if(!fp)return -1LL;
    ll ret=0;
    cc=0;
    while(~fscanf(fp,"%c",&tmp[cc]))++cc;
    fclose(fp);
    for(int i=0;i<cc;++i){
        bool ch=true;
        for(int j=0;ex[j];++j){
            if(tmp[i+j]!=ex[j]){
                ch=false;
                break;
            }
        }
        if(ch){
            i+=strlen(ex);
            while(tmp[i]==' '||tmp[i]==','||(tmp[i]>='0'&&tmp[i]<='9')){
                if(tmp[i]>='0'&&tmp[i]<='9'){
                    ret=ret*10+tmp[i]-48;
                }
                ++i;
            }
            return ret;
        }
    }
    return -2LL;
}
int main(){
    int en=0;
    FILE *ff=fopen("ex.txt","r");
    fscanf(ff,"%*c%*c%*c");
    while(~fscanf(ff,"%c",&ex[en]))printf("%d\n",ex[en]),++en;
    fclose(ff);
    FILE *fp=fopen("word=pre1.txt","r");
    FILE *f=fopen("result.txt","w");
    char t;
    int cnt=0;
    fscanf(fp,"%*c%*c%*c");
    while(fgets(s,1000,fp)){
        if(!ma[s]){
            ma[s]=true;
            s[strlen(s)-1]=0;
            fprintf(f,"%s ",s);
            for(int i=0;s[i];++i)if(s[i]==95)s[i]=32;
            fprintf(f,"%lld %lld\n",getgg(),getggxs());
            ++cnt;
            printf("%d\n",cnt);
        }
    }
    fclose(f);
    fclose(fp);
}
