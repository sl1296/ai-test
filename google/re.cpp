#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
char tt[1000],td[1000];
struct node{
    bool vis;
    int id,ln;
    string s;
    ll a,b;
    double x;
    vector<int> f,so;
    vector<vector<int> > path;
    void in(FILE *f){
        fscanf(f,"%s%lld%lld",tt,&a,&b);
        s=tt;
        x=(double)a/b;
    }
    void pri(){
        cout<<s<<endl;
        cout<<a<<endl;
        cout<<b<<endl;
        cout<<x<<endl;
    }
    bool operator < (const node &p) const{
        if(x!=p.x)return x>p.x;
        return b>p.b;
    }
    void getpath(node *ra,int &rcnt){
        for(int i=0;i<(int)f.size();++i){
            if(ra[f[i]].path.size()==0)ra[f[i]].getpath(ra,rcnt);
            for(int j=0;j<(int)ra[f[i]].path.size();++j){
                path.push_back(ra[f[i]].path[j]);
                path.back().push_back(id);
            }
        }
        if(path.size()==0){
            vector<int> tt;
            tt.push_back(id);
            path.push_back(tt);
        }
    }
    void out(FILE *f,node *ra,int &rcnt){
        fprintf(f,"ID%04d %.9f %s %lld %lld\n\n",id,x,s.c_str(),a,b);
        if(path.size()==0)getpath(ra,rcnt);
        for(int i=0;i<(int)path.size();++i){
            fprintf(f,"    ");
            for(int j=0;j<(int)path[i].size();++j){
                if(j)fprintf(f," > ");
                fprintf(f,"%d",path[i][j]);
            }
            fprintf(f,"\n    ");
            for(int j=0;j<(int)path[i].size();++j){
                if(j)fprintf(f," > ");
                fprintf(f,"%s",ra[path[i][j]].s.c_str());
            }
            fprintf(f,"\n\n");
        }
        fprintf(f,"--------------------------------------------\n");
    }
    bool have(int x){
        for(int i=0;i<(int)path.size();++i){
            for(int j=0;j<(int)path[i].size();++j){
                if(path[i][j]==x)return true;
            }
        }
        return false;
    }
};
int xf(char *s,node *ra,int &rcnt){
    string tmp=s;
    for(int i=0;i<rcnt;++i){
        if(ra[i].s==tmp)return i;
    }
    return -1;
}
void dfs(int pos,node *ra,int &rcnt){
    ra[pos].vis=true;
    if(ra[pos].so.size()==0){
        ra[pos].ln=1;
        return;
    }
    for(int i=0;i<(int)ra[pos].so.size();++i){
        if(!ra[ra[pos].so[i]].vis){
            dfs(ra[pos].so[i],ra,rcnt);
        }
        ra[pos].ln+=ra[ra[pos].so[i]].ln;
    }
}
bool cmp(node &a,node &b){

}
void readtree(node *ra,int &rcnt,char *s){
    FILE *f=fopen(s,"r");
    fscanf(f,"%*c%*c%*c%*s%*s%*s");
    while(~fscanf(f,"%s%s%*s",tt,td)){
//        printf("%s %s %d %d %d\n",tt,td,tt[0],td[0],strcmp(tt,"Artificial_intelligence"));
        int et=xf(tt,ra,rcnt);
        if(et==-1)et=rcnt,ra[rcnt].id=rcnt,ra[rcnt++].s=tt;
        int ef=xf(td,ra,rcnt);
        if(ef==-1)ef=rcnt,ra[rcnt].id=rcnt,ra[rcnt++].s=td;
        if(et!=ef){
            ra[et].f.push_back(ef);
            sort(ra[et].f.begin(),ra[et].f.end());
            ra[et].f.resize(unique(ra[et].f.begin(),ra[et].f.end())-ra[et].f.begin());
            ra[ef].so.push_back(et);
            sort(ra[ef].so.begin(),ra[ef].so.end());
            ra[ef].so.resize(unique(ra[ef].so.begin(),ra[ef].so.end())-ra[ef].so.begin());
            sort(ra[et].f.begin(),ra[et].f.end(),cmp);
            sort(ra[ef].so.begin(),ra[ef].so.end(),cmp);
        }
    }
    fclose(f);
}
void cmptree(node *ra,int ca,int rta,node *rb,int cb,int rtb){
    for(int i=)
}
#define N 28000
node ra[N],rb[N];
int ca,cb;
int main(){
    readtree(ra,ca,"tree1n.txt");
    readtree(rb,cb,"tree2.txt");
    cmptree(ra,ca,xf("Artificial_intelligence",ra,ca),rb,cb,xf("Artificial_intelligence",rb,cb));
}
