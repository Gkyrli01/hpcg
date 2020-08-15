//
// Created by kyrlitsias on 10/06/2020.
//

#ifndef HPCG_SYCLRESOURCES_H
#define HPCG_SYCLRESOURCES_H

#include <CL/sycl.hpp>
#include <map>
#include "Geometry.hpp"
#include "SyCLBufferContainer.h"


template<typename T,typename T1, int dims>
void GetBuffer(T arr, sycl::range<dims> range);

extern cl::sycl::gpu_selector neo;
extern sycl::queue queue;

extern BufferFactory bufferFactory;
extern BufferFactory dotFactory;
extern bool doAccess;
extern std::map<int, double *> dotProductArrays;
extern bool dotAccess;
extern  bool goEasyOnFwd;
extern bool using_reordering;

extern SyCLBufferContainer<local_int_t **, local_int_t, 2> *integerBuffers2D;
extern SyCLBufferContainer<local_int_t *, local_int_t, 1> *integerBuffers1D;
extern SyCLBufferContainer<char *, char, 1> *charBuffers1D;
extern SyCLBufferContainer<double **, double, 2> *doubleBuffers2D;
extern SyCLBufferContainer<double *, double, 1> *doubleBuffers1D;

//
//
//void SyCLCopyVector(Vector_STRUCT &x, Vector_STRUCT &to);
//
//void SyCLZeroVector(Vector_STRUCT &x);
//
//







//template<typename T,typename T1, int dims,sycl::access::mode Mode>
//sycl::accessor<T1,dims,Mode> GetAccessor(T arr, sycl::range<dims> range) {
//
//	if (strcmp(typeid(T).name(), "char") == 1) {
//		return charBuffers1D->GetBuffer(arr, range)->template get_access<Mode>();
//	} else if (typeid(T).name(), "double") {
//		if (dims == 2) {
//			return doubleBuffers2D->GetBuffer(arr, range)->template get_access<Mode>();
//		} else {
//			return doubleBuffers1D->GetBuffer(arr, range)->template get_access<Mode>();
//		}
//	} else {
//		if (dims == 2) {
//			return integerBuffers2D->GetBuffer(arr, range)->template get_access<Mode>();
//		} else {
//			return integerBuffers1D->GetBuffer(arr, range)->template get_access<Mode>();
//		}
//	}
//}

//sycl::buffer<char, 1> *GetBuffer(char *arr, sycl::range<1> range);


//void SyncBuffers();


//sycl::queue GetQueue(){
////	if(queue==nullptr){
////		auto exception_handler = [&](cl::sycl::exception_list eList) {
////			for (std::exception_ptr const &e : eList) {
////				try {
////					std::rethrow_exception(e);
////				} catch (cl::sycl::exception const &e) {
////					std::cout << "Failure" << std::endl;
////					std::terminate();
////				}
////			}
////		};
////		*queue=sycl::queue(neo,exception_handler);
////	}
//	return queue;
//}


#endif //HPCG_SYCLRESOURCES_H
