#ifndef NNET_TYPES_H_
#define NNET_TYPES_H_

#include <assert.h>
#include <cstddef>
#include <cstdio>

namespace nnet {

// Fixed-size array
template <typename T, unsigned N> struct array {
    typedef T value_type;
    static const unsigned size = N;

    T data[N];

    T &operator[](size_t pos) { return data[pos]; }

    const T &operator[](size_t pos) const { return data[pos]; }

    array &operator=(const array &other) {
        if (&other == this)
            return *this;

        assert(N == other.size && "Array sizes must match.");

        for (unsigned i = 0; i < N; i++) {
            #pragma HLS UNROLL
            data[i] = other[i];
        }
        return *this;
    }
};

template <typename T, unsigned M, unsigned N>
struct array2d {
    typedef T value_type;
    static const unsigned rows = M;
    static const unsigned cols = N;

    T data[M][N];

    T* operator[](size_t row) { return data[row]; }

    const T* operator[](size_t row) const { return data[row]; }

    array2d& operator=(const array2d& other) {
        if (&other == this)
            return *this;

        assert(M == other.rows && N == other.cols && "Array dimensions must match.");

        for (unsigned i = 0; i < M; i++) {
            for (unsigned j = 0; j < N; j++) {
                #pragma HLS UNROLL
                data[i][j] = other.data[i][j];
            }
        }
        return *this;
    }
};
/*
template <typename T, unsigned N, unsigned M> struct array2d {
    typedef T value_type;
    static const unsigned size1 = N;
    static const unsigned size2 = M;


    T data[N][M];

    T &operator[][](size_t pos1, size_t pos2) { return data[pos1][pos2]; }

    const T &operator[][](size_t pos1, size_t pos2) const { return data[pos1][pos2]; }

    array2d &operator=(const array2d &other) {
        if (&other == this)
            return *this;

        assert(N*M == other.size && "Array sizes must match.");

        for (unsigned i = 0; i < N; i++) {
            #pragma HLS UNROLL
            for (unsigned j = 0; j < M; j++) {
                data[i][j] = other[i][j];
            }
        }
        return *this;
    }
};
*/
// Generic lookup-table implementation, for use in approximations of math functions
template <typename T, unsigned N, T (*func)(T)> class lookup_table {
  public:
    lookup_table(T from, T to) : range_start(from), range_end(to), base_div(ap_uint<16>(N) / T(to - from)) {
        T step = (range_end - range_start) / ap_uint<16>(N);
        for (size_t i = 0; i < N; i++) {
            T num = range_start + ap_uint<16>(i) * step;
            T sample = func(num);
            samples[i] = sample;
        }
    }

    T operator()(T n) const {
        int index = (n - range_start) * base_div;
        if (index < 0)
            index = 0;
        else if (index > N - 1)
            index = N - 1;
        return samples[index];
    }

  private:
    T samples[N];
    const T range_start, range_end;
    ap_fixed<20, 16> base_div;
};

} // namespace nnet

#endif
