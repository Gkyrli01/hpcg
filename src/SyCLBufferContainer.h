//
// Created by kyrlitsias on 17/06/2020.
//

#ifndef DPC___SYCLBUFFERCONTAINER_H
#define DPC___SYCLBUFFERCONTAINER_H

#include <map>
#include <CL/sycl.hpp>

template<typename T1, int dims>
class BufferHolder {
public:
	sycl::buffer<T1, dims> *holder;

	BufferHolder(T1 *arr, sycl::range<dims> range) {
		holder = new sycl::buffer<T1, dims>(arr, range);
	}

};


template<typename T, typename T1, int dims>
class SyCLBufferContainer {
public:
	sycl::buffer<T1, dims> *GetBuffer(T1 **arr, sycl::range<dims> range) {
		if (map.find(arr) == map.end()) {
//			std::cout << "Not found existing buffer two dimensions" << std::endl;
			auto *b = new sycl::buffer<T1, dims>(*arr, range);
			map.insert(std::pair<T, sycl::buffer<T1, dims> *>(arr, b));
		}
		return map[arr];
	}

	std::map<T, sycl::buffer<T1, dims> *> map;

	sycl::buffer<T1, dims> *GetBuffer(T1 *arr, sycl::range<dims> range) {
		if (map.find(arr) == map.end()) {
			std::cout << "Not found existing buffer one dimension" << std::endl;
			auto *b = new sycl::buffer<T1, dims>(arr, range);
			map.insert(std::pair<T, sycl::buffer<T1, dims> *>(arr, b));
		}
		return map[arr];
	}
};


class BufferFactory {
public:
	SyCLBufferContainer<local_int_t **, local_int_t, 2> *integerBuffers2D = new SyCLBufferContainer<local_int_t **, local_int_t, 2>();
	SyCLBufferContainer<local_int_t *, local_int_t, 1> *integerBuffers1D = new SyCLBufferContainer<local_int_t *, local_int_t, 1>();
	SyCLBufferContainer<char *, char, 1> *charBuffers1D = new SyCLBufferContainer<char *, char, 1>();
	SyCLBufferContainer<double **, double, 2> *doubleBuffers2D = new SyCLBufferContainer<double **, double, 2>();
	SyCLBufferContainer<double *, double, 1> *doubleBuffers1D = new SyCLBufferContainer<double *, double, 1>();

	sycl::buffer<double, 2> *GetBuffer(double **arr, sycl::range<2> range) {
		return doubleBuffers2D->GetBuffer(arr, range);
	}

	sycl::buffer<local_int_t, 2> *GetBuffer(local_int_t **arr, sycl::range<2> range) {
		return integerBuffers2D->GetBuffer(arr, range);
	}

	sycl::buffer<char, 1> *GetBuffer(char *arr, sycl::range<1> range) {
		return charBuffers1D->GetBuffer(arr, range);
	}

	sycl::buffer<double, 1> *GetBuffer(double *arr, sycl::range<1> range) {
		return doubleBuffers1D->GetBuffer(arr, range);
	}

	sycl::buffer<local_int_t, 1> *GetBuffer(local_int_t *arr, sycl::range<1> range) {
		return integerBuffers1D->GetBuffer(arr, range);
	}

	void SyncBuffers(){
		//
		for (auto it = integerBuffers1D->map.begin(); it != integerBuffers1D->map.end(); ++it)
		{
			delete it->second;
		}
		for (auto it = integerBuffers2D->map.begin(); it != integerBuffers2D->map.end(); ++it)
		{
			delete it->second;
		}
		for (auto it = doubleBuffers1D->map.begin(); it != doubleBuffers1D->map.end(); ++it)
		{
			delete it->second;
		}
		for (auto it = doubleBuffers2D->map.begin(); it != doubleBuffers2D->map.end(); ++it)
		{
			delete it->second;
		}
		for (auto it = charBuffers1D->map.begin(); it != charBuffers1D->map.end(); ++it)
		{
			delete it->second;
		}
		integerBuffers1D->map.clear();
		integerBuffers2D->map.clear();
		doubleBuffers1D->map.clear();
		doubleBuffers2D->map.clear();
		charBuffers1D->map.clear();

	}

};

#endif //DPC___SYCLBUFFERCONTAINER_H
