//
// Created by kyrlitsias on 10/06/2020.
//
#include <map>
#include "SyCLResources.h"

cl::sycl::gpu_selector neo;
sycl::queue queue(neo);
std::map<int, double *> dotProductArrays;

BufferFactory bufferFactory;
BufferFactory dotFactory;

bool doAccess= true;
bool dotAccess= true;

SyCLBufferContainer<local_int_t **, local_int_t, 2> *integerBuffers2D= new SyCLBufferContainer<local_int_t **, local_int_t, 2>();
SyCLBufferContainer<local_int_t *, local_int_t, 1> *integerBuffers1D=new SyCLBufferContainer<local_int_t *, local_int_t, 1>();
SyCLBufferContainer<char *, char, 1> *charBuffers1D=new SyCLBufferContainer<char *, char, 1> ();
SyCLBufferContainer<double **, double, 2> *doubleBuffers2D=new SyCLBufferContainer<double **, double, 2>();
SyCLBufferContainer<double *, double, 1> *doubleBuffers1D=new SyCLBufferContainer<double *, double, 1>();


//template<typename T,typename T1, int dims>
//sycl::buffer<T1,dims>* GetBuffer(T arr, sycl::range<dims> range){
//
//	if (strcmp(typeid(T1).name(), "char") == 1) {
//		return charBuffers1D->GetBuffer(arr, range);
//	} else if (typeid(T1).name(), "double") {
//		if (dims == 2) {
//			return doubleBuffers2D->GetBuffer(arr, range);
//		} else {
//			return doubleBuffers1D->GetBuffer(arr, range);
//		}
//	} else {
//		if (dims == 2) {
//			return integerBuffers2D->GetBuffer(arr, range);
//		} else {
//			return integerBuffers1D->GetBuffer(arr, range);
//		}
//	}
//}

template<typename T,typename T1, int dims>
void GetBuffer(T arr, sycl::range<dims> range){

	if (strcmp(typeid(T1).name(), "char") == 1) {
		std::cout<<"ischar"<<std::endl;
	} else if (typeid(T1).name(), "double") {
		std::cout<<"double"<<std::endl;
	} else {
		std::cout<<"int"<<std::endl;

	}
}
//
//void SyCLCopyVector(Vector_STRUCT &x, Vector_STRUCT &to) {
//	local_int_t size = x.localLength;
//	auto x_buf = *x.buf;
//	auto to_buf = *to.buf;
//	{
//		queue.submit([&](sycl::handler &cgh) {
//			auto to_acc = to_buf.get_access<sycl::access::mode::write>(cgh);
//			auto x_acc = x_buf.get_access<sycl::access::mode::read>(cgh);
//			cgh.parallel_for<class copy>(
//					sycl::nd_range<1>(size, 32),
//					[=](sycl::nd_item<1> item) {
//						size_t i = item.get_global_linear_id();
//						if (i < size)
//							to_acc[i] = x_acc[i];
//					});
//		});
//	}
//	auto access = to_buf.get_access<sycl::access::mode::read>();
//}
//
//void SyCLZeroVector(Vector_STRUCT &x) {
//	local_int_t size = x.paddedLength;
//	auto x_buf = *x.buf;
//	{
//
//		queue.submit([&](sycl::handler &cgh) {
//			auto x_acc = x_buf.get_access<sycl::access::mode::write>(cgh);
//			cgh.parallel_for<class zero>(
//					sycl::nd_range<1>(size, 32),
//					[=](sycl::nd_item<1> item) {
//						size_t i = item.get_global_linear_id();
//						if (i < size)
//							x_acc[i] = 0;
//					});
//		});
//	}
//	x_buf.get_access<sycl::access::mode::read>();
//}


//void SyncBuffers(){
//	bufferFactory.SyncBuffers();
////
////	for (auto it = integerBuffers1D->map.begin(); it != integerBuffers1D->map.end(); ++it)
////	{
////		it->second->get_access<sycl::access::mode::read>();
////	}
////	for (auto it = integerBuffers2D->map.begin(); it != integerBuffers2D->map.end(); ++it)
////	{
////		it->second->get_access<sycl::access::mode::read>();
////	}
////	for (auto it = doubleBuffers1D->map.begin(); it != doubleBuffers1D->map.end(); ++it)
////	{
////		it->second->get_access<sycl::access::mode::read>();
////	}
////	for (auto & it : doubleBuffers2D->map)
////	{
////		it.second->get_access<sycl::access::mode::read>();
////	}
////	for (auto it = charBuffers1D->map.begin(); it != charBuffers1D->map.end(); ++it)
////	{
////		it->second->get_access<sycl::access::mode::read>();
////	}
//}