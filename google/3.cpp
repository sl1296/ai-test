#include<bits/stdc++.h>
using namespace std;
struct node{
    string s;
    long long a,b;
    double x;
    bool operator < (const node &p) const{
        if(x!=p.x)return x>p.x;
        else return b>p.b;
    }
};
char s[1000];
node m[3000];
int main(){
    FILE *f=fopen("result.txt","r");
    FILE *d=fopen("ret.txt","w");
    for(int i=0;i<2079;++i){
        fscanf(f,"%s%lld%lld",s,&m[i].a,&m[i].b);
        m[i].s=s;
        m[i].x=(double)m[i].a/m[i].b;
    }
    sort(m,m+2079);
    for(int i=0;i<2079;++i){
        fprintf(d,"%.6f %s %lld %lld\n",m[i].x,m[i].s.c_str(),m[i].a,m[i].b);
    }
    fclose(f);
    fclose(d);
}
