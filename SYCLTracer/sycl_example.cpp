// https://github.com/jeffhammond/dpcpp-tutorial
#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <exception>

using namespace sycl;

constexpr int N = 42;

int main(){
	std::array<int, N> a, b;
	for(int i=0; i< N; ++i) {
		a[i] = i;
		b[i] = 0;
	}
	
	try {
		queue Q(host_selector{}); 
		buffer<int, 1> A{a.data(), range<1>(a.size())};
		buffer<int, 1> B{b.data(), range<1>(b.size())};
		
		Q.submit([&](handler& h) {
			auto accA = A.template get_access<access::mode::read>(h);
			auto accB = B.template get_access<access::mode::write>(h);
			h.parallel_for<class nstream>(
				range<1>{N},
				[=](id<1> i) { accB[i] = accA[i]; });
			
		});
		Q.wait();
		B.get_access<access::mode::read>(); // <--- Host Accessor to Synchronize Memory
		for(int i=0; i< N; ++i) {
			std::cout << b[i] << " ";
		}
	}
	catch(sycl::exception& ex)
	{
		std::cerr << "SYCL Exception thrown: " << ex.what() << std::endl;
	}
	catch(std::exception& ex)
	{
		std::cerr << "std Exception thrown: " << ex.what() << std::endl;
	}
	std::cout << "\nDone!\n";
	return 0;
}
