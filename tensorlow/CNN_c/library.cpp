#include <cstdio>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <thread>
#include <cblas.h>
using std::cout;
using std::endl;
typedef unsigned int uint;
const int thread_num = 8;


uint num, n, m, fn, fm, fin, fout, nn, mm, sn, sm, new_n, new_m;
float *img, *flt, *result, *grad;
int *maxpos;

inline void matmul(float* a, float* b, float* c, uint n, uint k, uint m, float beta = 0){
//    cblas_sgemm(order,transA,transB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);
//    alpha =1,beta =0 的情况下，等于两个矩阵相成。
//    第一参数 oreder 候选值 有ClasRowMajow 和ClasColMajow 这两个参数决定一维数组怎样存储在内存中,一般用ClasRowMajow
//    参数 transA和transB ：表示矩阵A，B是否进行转置。候选参数 CblasTrans 和CblasNoTrans.
//    参数M：表示 A或C的行数。如果A转置，则表示转置后的行数
//    参数N：表示 B或C的列数。如果B转置，则表示转置后的列数。
//    参数K：表示 A的列数或B的行数（A的列数=B的行数）。如果A转置，则表示转置后的列数。
//    参数LDA：表示A的列数，与转置与否无关。
//    参数LDB：表示B的列数，与转置与否无关。
//    参数LDC：始终=N
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, m, k, (float)1, a, k, b, m, beta, c, m);
}

inline uint imgpos(const uint &i, const uint &j = 0, const uint &p = 0, const uint &q = 0){
    return q + fin * (p + m * (j + i * n));
}

inline uint fltpos(const uint &i, const uint &j = 0, const uint &p = 0, const uint &q = 0){
    return q + fout * (p + fin * (j + i * fm));
}

inline uint respos(const uint &i, const uint &j = 0, const uint &p = 0, const uint &q = 0){
    return q + fout * (p + m * (j + i * n));
}


inline uint polpos(const uint &i, const uint &j = 0, const uint &p = 0, const uint &q = 0){
    return q + fout * (p + mm * (j + i * nn));
}

void flip_filter(float *flt, float *res, int n, int m, int fin, int fout){
    int _size = fin * fout;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            int x = n - i - 1, y = m - j - 1;
            memcpy(res + (x * m + y) * _size, flt + (i * m + j) * _size, sizeof(float) * _size);
        }
    }
}

void trans_in_out(float* &filter, uint n, uint m, uint fin, uint fout){
    float *result = (float*)malloc(n * m * fin * fout * sizeof(float));
    int _size = fin * fout;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            float* res = result + _size * (j + m * i);
            float* flt = filter + _size * (j + m * i);
            for(int p = 0; p < fin; p++)
                for(int q = 0; q < fout; q++)
                    res[p + q * fin] = flt[q + p * fout];
        }
    }
    free(filter);
    filter = result;
}

const uint memory_size = (4 * 28 * 28 * 5 * 5 * 32);
float image[memory_size];

void set_zero(uint num, const int &new_n, const int &new_m,
              const int &u, const int &d, const int &l, const int &r, const int &last){
    int row = fn * fm * last;
    for(uint now = 0, _size = sizeof(float) * last; now < num; now++){
        float* mat = image + (now * new_n * new_m * row);
        //clo2im
        for (int i = u; i < d; i++){
            for (int j = l; j < r; j++){
                int td = i + fn, tr = j + fm;
                if(i >= 0 && j >= 0 && td < n && tr < m)
                    continue;
                float* tmp_now = mat + row * ((j - l) + (i - u) * m);
                for (int x = i; x < td; x++){
                    for (int y = j; y < tr; y++, tmp_now += last){
                        if(x < 0 || y < 0 || x >= n || y >= m)
                            memset(tmp_now, 0, _size);
                    }
                }
            }
        }
    }
}

void run_conv2d(uint L, uint R, const int &new_n, const int &new_m, const int &u, const int &d,
                const int &l, const int &r, const uint &last, const uint &out){
    int row = fn * fm * last;
    for(uint now = L, _size = sizeof(float) * last; now < R; now++){
        float* img_now = img + imgpos(now);
        float* mat = image + ((now - L) * new_n * new_m * row);
        //clo2im
        for (int i = u; i < d; i++){
            for (int j = l; j < r; j++){
                float* tmp_now = mat + row * ((j - l) + (i - u) * m);
                int td = i + fn, ll = std::max(j, 0), rr = std::min(j + fm, m);
                uint tmp_add = (ll - j) * last, len = _size * (rr - ll);
                for (int x = i; x < td; x++, tmp_now += last * fm){
                    if(x < 0 || x >= n)
                        continue;
                    memcpy(tmp_now + tmp_add, img_now + (x * m + ll) * last, len);
                }
            }
        }
    }
    float* res = result + (L * new_n * new_m * out);
    matmul(image, flt, res, (uint)((R - L) * new_n * new_m), fn * fm * last, out);
}


extern "C"
void conv2d(float *_img, float *_flt, float *_result,
          int _num, int _n, int _m, int _fn, int _fm, int _fin, int _fout, bool same) {
    img = _img;
    flt = _flt;
    result = _result;
    num = (uint)_num;
    n = (uint)_n;
    m = (uint)_m;
    fn = (uint)_fn;
    fm = (uint)_fm;
    fin = (uint)_fin;
    fout = (uint)_fout;
    int l = 0, r = m, u = 0, d = n;
    if(same)
        l = -(_fm - 1) / 2, r = _m + _fm / 2, u = -(_fn - 1) / 2, d = _n + _fn / 2;
    d -= _fn - 1;
    r -= _fm - 1;
    int new_n = d - u, new_m = r - l;
    uint batch_size = memory_size / (new_n * new_m * fn * fm * fin), now = 0;
    set_zero(batch_size, new_n, new_m, l, r, u, d, fin);
    while(now < num){
        uint nex =  now + batch_size;
        if(nex > num)
            nex = num;
        run_conv2d(now, nex, new_n, new_m, l, r, u, d, fin, fout);
        now = nex;
    }
}


extern "C"
void backup_conv2d_image(float *_img, float *_flt, float *_result,
                         int _num, int _n, int _m, int _fn, int _fm, int _fin, int _fout, bool same) {
    num = (uint)_num;
    n = (uint)_n;
    m = (uint)_m;
    fn = (uint)_fn;
    fm = (uint)_fm;
    fin = (uint)_fin;
    fout = (uint)_fout;
    img = _img;
    flt = (float*)malloc(fn * fm * fin * fout * sizeof(float));
    flip_filter(_flt, flt, fn, fm, fin, fout);
    trans_in_out(flt, fn, fm, fin, fout);
    result = _result;

    int l, r, u, d;
    if(same)
        l = -_fm / 2, r = _m + (_fm - 1) / 2, u = -_fn / 2, d = _n + (_fn - 1) / 2;
    else
        l = -_fm + 1, r = _m + _fm - 1, u = -_fn + 1, d = _n + _fn - 1;
    d -= fn - 1;
    r -= fm - 1;
    int new_n = d - u, new_m = r - l;

    uint batch_size = memory_size / (new_n * new_m * fn * fm * fout), now = 0;
    set_zero(batch_size, new_n, new_m, l, r, u, d, fout);
    while(now < num){
        uint nex =  now + batch_size;
        if(nex > num)
            nex = num;
        run_conv2d(now, nex, new_n, new_m, l, r, u, d, fout, fin);
        now = nex;
    }
    free(flt);
}

extern "C"
void backup_conv2d_filter(float *_img, float *_flt, float *_result,
                          int _num, int _n, int _m, int _fn, int _fm, int _fin, int _fout, bool same) {
    num = (uint)_num;
    n = (uint)_n;
    m = (uint)_m;
    fin = (uint)_fin;
    fout = (uint)_fout;
    img = _img;
    flt = _flt;
    result = _result;

    int l = 0, r = m, u = 0, d = n;
    if(same){
        l = -_fm / 2, r = _m + (_fm - 1) / 2, u = -_fn / 2, d = _n + (_fn - 1) / 2;
        fn = n;
        fm = m;
    }
    else{
        fn = n - _fn + 1;
        fm = m - _fm + 1;
    }

    d -= fn - 1;
    r -= fm - 1;
    int new_n = d - u, new_m = r - l;
    memset(image, 0, new_n * new_m * fin * fn * fm * sizeof(float));

    for(uint now = 0; now < num; now++) {
        float *img_now = img + (now * n * m * fin);
        //clo2im
        for (int i = u; i < d; i++) {
            for (int j = l; j < r; j++) {
                float* tmp_img = image + fn * fm * fin * ((i - u) * new_m + j - l);
                int td = i + fn, ll = std::max(j, 0), rr = std::min(j + fm, m);
                for (int x = i; x < td; x++, tmp_img += fm){
                    if(x < 0 || x >= n)
                        continue;
                    float *tmp_img1 = tmp_img, *tmp_tmp_img, *img_now_now, *end_img;
                    for (int y = ll; y < rr; y++, tmp_img1++) {
                        tmp_tmp_img = tmp_img1;
                        img_now_now = img_now + (x * m + y) * fin;
                        end_img = img_now_now + fin;
                        for (; img_now_now != end_img; tmp_tmp_img += fn * fm, img_now_now++)
                            *tmp_tmp_img = *img_now_now;
                    }
                }
            }
        }
        float *filter = flt + (now * fn * fm * fout);
        matmul(image, filter, result, (uint)(new_n * new_m) * fin, fn * fm, fout, (float)(now > 0));
    }
}
void do_max_pool(uint l, uint r){
    for(uint now = l; now < r; now++){
        for(uint i = 0, idi = 0, tmp = 0; i < n; idi = ++i / sn){
            for(uint j = 0, idj = 0; j < m; idj = ++j / sm, tmp++){
                float *image = img + imgpos(now, i, j);
                float *res = result + polpos(now, idi, idj);
                int *pos = maxpos + polpos(now, idi, idj);
                for(uint k = 0; k < fout; k++, res++, image++){
                    if(*image > *res){
                        *res = *image;
                        pos[k] = tmp;
                    }
                }
            }
        }
    }
}


extern "C"
void max_pool(float *_img, float *_result, int *_maxpos,
              int _num, int _n, int _m, int _sn, int _sm, int _fout){
    img = _img;
    result = _result;
    maxpos = _maxpos;
    num = (uint)_num;
    n = (uint)_n;
    m = (uint)_m;
    sn = (uint)_sn;
    sm = (uint)_sm;
    fin = fout = (uint)_fout;
    nn = n / _sn;
    mm = m / _sm;
    memset(result, 254, num * nn * mm * fout * sizeof(float));
    int k = num / thread_num, rest = num % thread_num;
    int nums[8] = {k, k, k, k, k, k, k, k};
    for(int i = 0; i < rest; i++)
        nums[i]++;

    std::thread th[8];
    th[0] = std::thread(do_max_pool, 0, nums[0]);
    for(int i = 1; i < thread_num; i++){
        nums[i] += nums[i - 1];
        th[i] = std::thread(do_max_pool, nums[i - 1], nums[i]);
    }
    for(int i = 0; i < thread_num; i++)
        th[i].join();
}

void do_backup_max_pool(uint l, uint r){
    for(uint now = l; now < r; now++){
        float *res = result + imgpos(now);
        for(uint i = 0, idi = 0, tmp = 0; i < n; idi = ++i / sn){
            for(uint j = 0, idj = 0; j < m; idj = ++j / sm, tmp++){
                float *gra = grad + polpos(now, idi, idj);
                int *pos = maxpos + polpos(now, idi, idj);
                for(uint k = 0; k < fout; k++, pos++, res++, gra++){
                    if(tmp == *pos)
                        *res = *gra;
                    else
                        *res = 0;
                }
            }
        }
    }
}

extern "C"
void backup_max_pool(int *_maxpos, float *_grad, float *_result,
              int _num, int _n, int _m, int _sn, int _sm, int _fout){
    maxpos = _maxpos;
    grad = _grad;
    result = _result;
    num = (uint)_num;
    n = (uint)_n;
    m = (uint)_m;
    fin = fout = (uint)_fout;
    nn = n / _sn;
    mm = m / _sm;
    sn = (uint)_sn;
    sm = (uint)_sm;
    int k = num / thread_num, rest = num % thread_num;
    int nums[8] = {k, k, k, k, k, k, k, k};

    for(int i = 0; i < rest; i++)
        nums[i]++;

    std::thread th[8];
    th[0] = std::thread(do_backup_max_pool, 0, nums[0]);
    for(int i = 1; i < thread_num; i++){
        nums[i] += nums[i - 1];
        th[i] = std::thread(do_backup_max_pool, nums[i - 1], nums[i]);
    }
    for(int i = 0; i < thread_num; i++)
        th[i].join();
}

extern "C"
void Matmul(float* a, float* b, float* c, int na, int ma, int nb, int mb, bool transA, bool transB){
    if(transA){
        if(transB)
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, ma, nb, mb, (float)1, a, ma, b, mb, 0, c, nb);
        else
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ma, mb, na, (float)1, a, ma, b, mb, 0, c, mb);
    }
    else{
        if(transB)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, na, nb, mb, (float)1, a, ma, b, mb, 0, c, nb);
        else
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, na, mb, ma, (float)1, a, ma, b, mb, 0, c, mb);
    }
}

void run_sgn(float* a, float* result, int n){
    float* end = a + n;
    for(;a !=end; a++, result++)
        *result = ((*a) > 0);
}

extern "C"
void sgn(float* a, float* result, int n){
    if(n > 2000000){
        int k = n / thread_num, rest = n % thread_num;
        int nums[8] = {k, k, k, k, k, k, k, k};

        for(int i = 0; i < rest; i++)
            nums[i]++;

        std::thread th[8];
        th[0] = std::thread(run_sgn, a, result, nums[0]);
        for(int i = 1; i < thread_num; i++){
            a += nums[i - 1];
            result += nums[i - 1];
            th[i] = std::thread(run_sgn, a, result, nums[i]);
        }
        for(int i = 0; i < thread_num; i++)
            th[i].join();
    }
    else {
        float *end = a + n;
        for (; a != end; a++, result++)
            *result = ((*a) > 0);
    }

}

void run_relu(float* a, float* result, int n){
    float *end = a + n;
    for (; a != end; a++, result++)
        *result = ((*a) > 0) ? *a : 0;
}

extern "C"
void relu(float* a, float* result, int n){
    if(n > 2000000){
        int k = n / thread_num, rest = n % thread_num;
        int nums[8] = {k, k, k, k, k, k, k, k};

        for(int i = 0; i < rest; i++)
            nums[i]++;

        std::thread th[8];
        th[0] = std::thread(run_relu, a, result, nums[0]);
        for(int i = 1; i < thread_num; i++){
            a += nums[i - 1];
            result += nums[i - 1];
            th[i] = std::thread(run_relu, a, result, nums[i]);
        }
        for(int i = 0; i < thread_num; i++)
            th[i].join();
    }
    else {
        float *end = a + n;
        for (; a != end; a++, result++)
            *result = ((*a) > 0) ? *a : 0;
    }
}
