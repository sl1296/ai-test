#include<cstdio>
#include<cstring>
char a[1000];
int main(){
    FILE *f=fopen("F:\\test\\keywords.txt","r"),*e=fopen("F:\\test\\out.txt","w");
    int cc=0;
    while(fgets(a,1000,f)){
        if(cc%5==0){
            if(cc)fprintf(e,")\n");
            fprintf(e,"Find(");
        }
        a[strlen(a)-1]=0;
        fprintf(e,"\"%s\"",a);
        if(cc%5!=4)fprintf(e,",");
        ++cc;
    }
    while(cc%5!=0){
        fprintf(e,"1");
        if(cc%5!=4)fprintf(e,",");
        ++cc;
    }
    fprintf(e,")\n");
    fclose(f);
    fclose(e);
    return 0;
}
