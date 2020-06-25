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


void SyncBuffers(){
	bufferFactory.SyncBuffers();
//
//	for (auto it = integerBuffers1D->map.begin(); it != integerBuffers1D->map.end(); ++it)
//	{
//		it->second->get_access<sycl::access::mode::read>();
//	}
//	for (auto it = integerBuffers2D->map.begin(); it != integerBuffers2D->map.end(); ++it)
//	{
//		it->second->get_access<sycl::access::mode::read>();
//	}
//	for (auto it = doubleBuffers1D->map.begin(); it != doubleBuffers1D->map.end(); ++it)
//	{
//		it->second->get_access<sycl::access::mode::read>();
//	}
//	for (auto & it : doubleBuffers2D->map)
//	{
//		it.second->get_access<sycl::access::mode::read>();
//	}
//	for (auto it = charBuffers1D->map.begin(); it != charBuffers1D->map.end(); ++it)
//	{
//		it->second->get_access<sycl::access::mode::read>();
//	}
}