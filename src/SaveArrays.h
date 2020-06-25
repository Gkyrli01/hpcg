//
// Created by kyrlitsias on 01/06/2020.
//
#include <fstream>


#ifndef HPCG_SAVEARRAYS_H
#define HPCG_SAVEARRAYS_H

#endif //HPCG_SAVEARRAYS_H

template <class T>
bool saveArray(const T *pdata, size_t length, const std::string &file_path) {
	std::ofstream os(file_path.c_str(), std::ios::binary | std::ios::out);
	if (!os.is_open())
		return false;
	os.write(reinterpret_cast<const char *>(pdata), std::streamsize(length * sizeof(T)));
	os.close();
	return true;
}
template <class T>
bool save2dArray(T **pdata, size_t rows, size_t elements, const std::string &file_path) {
	std::ofstream os(file_path.c_str(), std::ios::binary | std::ios::out);
	if (!os.is_open())
		return false;
	for (size_t i = 0; i < rows; ++i) {
		os.write(reinterpret_cast<const char *>(pdata[i]), std::streamsize(elements * sizeof(T)));
	}
	os.close();
	return true;
}
