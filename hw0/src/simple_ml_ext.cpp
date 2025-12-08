/**
 * 毫无疑问，这是一段不那么优雅的代码实践
 * 然而码的很爽
 * 1. tem 发生了变量遮蔽
 * 2. elems 与 size 用混了
 * 3. 指数爆炸风险
 * 课上看到H君分心了写出来的
 * 白天网卡死了，泪目
 * linux makefile与windows的不兼容，最终放弃windows jupeter，转用colab，
 * 研究了半天drive file怎么加载，ipynb怎么保存，泪目
 * 第二天又debug了一个小时，发现是有一段公共的赋值应该放在条件编译分支外面，气死了
 * 指数爆炸了，不知道为什么！！！
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <cstring>
#include <stdexcept>


//#define UGLY

#ifdef UGLY
#else
#include <algorithm>
#endif

//写着玩
#define MAX(ptr, num) do{}while()//唉， 算了



namespace py = pybind11;

// 这一段我自己非常喜欢，很优雅
void matrixmul(float* Xb, float* theta, float* result,
    size_t row_xb, size_t n, size_t k) //row_xb * n  , n * k
{
    for (int i = 0; i < row_xb; i++) {
        for (int ii = 0; ii < k; ii++) {    //这是对应的theta的列数k
            for (int iii = 0; iii < n; iii++) { //这才具体到每一个乘法运算
                result[i * k + ii] += Xb[i * n + iii] * theta[iii * k + ii];
            }
        }
    }
}

void matrixexp(float* result, size_t elems) {
    for(int i = 0; i < elems; i++) {
        result[i] = std::exp(result[i]);
    }
}


const float eps = 1e-12f;
void matrixexp_sum(float* row_sum, float* result, size_t k, size_t elems) {
    for (int i = 0; i < elems; i += k) {
        int iter = i/k;
        int s = k;
        while(--s >= 0) {
            row_sum[iter] += result[k*iter + s];
        }
        row_sum[iter] = std::max(row_sum[iter], eps);
    }
}

float* matrix_trans(float* Xb, size_t sizeXb, size_t n) {
    float* tem = (float*) malloc(sizeXb);
    int row_num = sizeXb/ (sizeof(float) * n);
    int col_num = n;
    for(int i = 0; i < row_num; i++) {
        for(int j = 0; j < col_num; j++) {
            tem[j * row_num + i] = Xb[i * n + j]; //tem[j][i] = Xb[i][j]
        }
    }
    return tem;
}



void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int iter = (m + batch - 1) / batch;

    for(int i = 0; i < iter; i++) {
        float *Xb,  *result, *grad;
        int *Yb;
        size_t sizeXb;
        size_t sizeYb;
        size_t sizeResult = m * k * sizeof(float);
        size_t sizeGrad = n * k * sizeof(float);
        int cur_batch_size = 0;
        //下面这个判断十分丑陋
        #ifdef UGLY
        if (i != iter - 1) {
            sizeXb = batch * n * sizeof(float);
            sizeYb = batch * sizeof(unsigned char);
            cur_batch_size = batch;
        }
        else{
            cur_batch_size = m - batch * i;
            sizeXb = (m - batch * i) * n * sizeof(float);
            sizeYb = (m - batch * i) * sizeof(float);
        }
        //一行解决问题！
        #else
        cur_batch_size = std::min(batch, m - batch * i);
        #endif
        sizeXb = cur_batch_size * n * sizeof(float);
        sizeYb = cur_batch_size * sizeof(int);
        sizeResult = cur_batch_size * k * sizeof(float);
        Xb = (float*) malloc(sizeXb);
        Yb = (int*) malloc(sizeYb);
        grad = (float*) malloc(sizeGrad);
        result = (float*) malloc(sizeResult);
        memset(result, 0, sizeResult);
        memcpy(Xb, &X[i * batch * n], sizeXb);
        //memcpy(Yb, y_xxx)
        for (int ii = 0; ii < cur_batch_size; ii++) {
            Yb[ii] = static_cast<int>(y[ii + i * batch]);
        }
        matrixmul(Xb, theta, result, cur_batch_size, n, k);
        #ifdef UGLY
        #else 
        //防止指数爆炸，耶耶耶
        for(int i = 0; i < cur_batch_size; i++) {
            float* start = result + i * k;
            float row_max = *std::max_element(start, start + k);
            for(int j = 0; j < k; j++)
                start[j] -= row_max;
        }
        #endif
        //notice in cmath , we use: std::exp, std::log (stands for ln),std::log2
        matrixexp(result, sizeResult / sizeof(float));

        float* row_sum = (float*) malloc(sizeResult / k);
        memset(row_sum, 0, sizeResult / k);
        matrixexp_sum(row_sum, result, k, sizeResult / sizeof(float));

        for (int i = 0; i < cur_batch_size; i++) {
            int tem = k;
            while(--tem >= 0) {
                // div row_sum
                result[i * k + tem] = result[i * k + tem] / row_sum[i];       
            }
            // minus Iy
            int index = k * i + Yb[i];
            result[k * i + Yb[i]] -= 1;
        }

        float* Xb_T = matrix_trans(Xb, sizeXb, n); // Xb_T: (n, batch_size)

        float* tem = (float*) malloc(n * k * sizeof(float));
        memset(tem, 0, n * k * sizeof(float));
        matrixmul(Xb_T, result, tem, n, cur_batch_size, k);
        memcpy(grad, tem, sizeGrad);
        //matrix minus 
 
        float constant = lr / cur_batch_size;
        int count = n * k;
        while(--count >= 0) {
            theta[count] -= constant * grad[count];
            if(std::isnan(theta[count])) {
              printf("%d <- the wrong batch_num",i);
              throw std::runtime_error("ohhhh!");
            }
        }

        //free
        free(Xb_T);
        free(Xb);
        free(Yb);
        free(grad);
        free(result);
        free(row_sum);
        free(tem);
    }

    /// END YOUR CODE       
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
