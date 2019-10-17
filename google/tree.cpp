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
    void getpath();
    void out(FILE *f);
    bool have(int x){
        for(int i=0;i<path.size();++i){
            for(int j=0;j<path[i].size();++j){
                if(path[i][j]==x)return true;
            }
        }
        return false;
    }
};
node ra[28000];
int rcnt;
void node::getpath(){
    for(int i=0;i<f.size();++i){
        if(ra[f[i]].path.size()==0)ra[f[i]].getpath();
        for(int j=0;j<ra[f[i]].path.size();++j){
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
void node::out(FILE *f){
    fprintf(f,"ID%04d %.9f %s %lld %lld\n\n",id,x,s.c_str(),a,b);
    if(path.size()==0)getpath();
    for(int i=0;i<path.size();++i){
        fprintf(f,"    ");
        for(int j=0;j<path[i].size();++j){
            if(j)fprintf(f," > ");
            fprintf(f,"%d",path[i][j]);
        }
        fprintf(f,"\n    ");
        for(int j=0;j<path[i].size();++j){
            if(j)fprintf(f," > ");
            fprintf(f,"%s",ra[path[i][j]].s.c_str());
        }
        fprintf(f,"\n\n");
    }
    fprintf(f,"--------------------------------------------\n");
}
int xf(char *s){
    string tmp=s;
    for(int i=0;i<rcnt;++i){
        if(ra[i].s==tmp)return i;
    }
    return -1;
}
void dfs(int pos){
    ra[pos].vis=true;
    if(ra[pos].so.size()==0){
        ra[pos].ln=1;
        return;
    }
    for(int i=0;i<ra[pos].so.size();++i){
        if(!ra[ra[pos].so[i]].vis){
            dfs(ra[pos].so[i]);
        }
        ra[pos].ln+=ra[ra[pos].so[i]].ln;
    }
}
int main(){
//    FILE *f=fopen("result.txt","r");
//    fscanf(f,"%*c%*c%*c");
//    for(int i=0;i<2079;++i){
//        ra[i].in(f);
//    }
//    fclose(f);
//    sort(ra,ra+2079);
//    for(int i=0;i<2079;++i)ra[i].id=i;
    FILE *f;
    f=fopen("tree.txt","r");
    fscanf(f,"%*c%*c%*c%*s%*s");
    while(~fscanf(f,"%s%s",tt,td)){
//        printf("in:%s %s\n",tt,td);
        int et=xf(tt);
        if(et==-1)et=rcnt,ra[rcnt].id=rcnt,ra[rcnt++].s=tt;
        int ef=xf(td);
        if(ef==-1)ef=rcnt,ra[rcnt].id=rcnt,ra[rcnt++].s=td;
        if(et!=ef){
            ra[et].f.push_back(ef);
            sort(ra[et].f.begin(),ra[et].f.end());
            ra[et].f.resize(unique(ra[et].f.begin(),ra[et].f.end())-ra[et].f.begin());
            ra[ef].so.push_back(et);
            sort(ra[ef].so.begin(),ra[ef].so.end());
            ra[ef].so.resize(unique(ra[ef].so.begin(),ra[ef].so.end())-ra[ef].so.begin());
        }
    }
    fclose(f);
    f=fopen("tree2.txt","r");
    fscanf(f,"%*c%*c%*c%*s%*s%*s");
    while(~fscanf(f,"%s%s%*s",tt,td)){
//        printf("in2:%s %s\n",tt,td);

        int et=xf(tt);
        if(et==-1)et=rcnt,ra[rcnt].id=rcnt,ra[rcnt++].s=tt;
        int ef=xf(td);
        if(ef==-1)ef=rcnt,ra[rcnt].id=rcnt,ra[rcnt++].s=td;
//        printf("%s %s\n",ra[rcnt-2].s.c_str(),ra[rcnt-1].s.c_str());
        if(et!=ef){
            ra[et].f.push_back(ef);
            sort(ra[et].f.begin(),ra[et].f.end());
            ra[et].f.resize(unique(ra[et].f.begin(),ra[et].f.end())-ra[et].f.begin());
            ra[ef].so.push_back(et);
            sort(ra[ef].so.begin(),ra[ef].so.end());
            ra[ef].so.resize(unique(ra[ef].so.begin(),ra[ef].so.end())-ra[ef].so.begin());
        }
    }
    fclose(f);
//    f=fopen("rt2.txt","w");
//    FILE *fp=fopen("addx.txt","w");
//    for(int i=0;i<2079;++i){
//        ra[i].vis=false;
//        ra[i].out(f);
//        if(!ra[i].have(449)){
//            fprintf(fp,"%f %s %lld %lld\n",ra[i].x,ra[i].s.c_str(),ra[i].a,ra[i].b);
//        }
//    }
//    fclose(f);
//    fclose(fp);
    int aa=0,bb=0;
    printf("%d %s\n",xf("Artificial_intelligence"),ra[0].s.c_str());
    for(int i=0;i<ra[0].f.size();++i){
        printf("%d %s\n",ra[0].f[i],ra[ra[0].f[i]].s.c_str());
    }
    dfs(xf("Artificial_intelligence"));
    printf("rcnt=%d\n",rcnt);
    for(int i=0;i<rcnt;++i){
        if(ra[i].f.size()==0)
        printf("--%d %s \n",i,ra[i].s.c_str());
    }
    while(true){
        int type,id;
        scanf("%d%d",&type,&id);
        if(type){
            for(int i=0;i<ra[id].so.size();++i){
                if(i%6==5)printf("\n");
                printf("%5d(%5d,%5d) ",ra[id].so[i],ra[ra[id].so[i]].so.size(),ra[ra[id].so[i]].ln);
            }
            printf("\n");
            for(int i=0;i<ra[id].so.size();++i){
                printf("%s\n",ra[ra[id].so[i]].s.c_str());
            }
        }else{
            printf("%d %d %s\n",ra[id].so.size(),ra[id].ln,ra[id].s.c_str());
        }
    }
}
