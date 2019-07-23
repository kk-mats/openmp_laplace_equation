#include <iostream>
#include <cmath>
#include <array>
#include <vector>
#include <fstream>
#include <execution>
#include <chrono>
#include <omp.h>

class cpu_potential_solver
{
public:
	cpu_potential_solver(const int h)
		: h_(h), buffer_(h, std::vector<float>(h, 0)), circles_({circle(0.25*h, 0.75*h, 100, 0.125*h), circle(0.875*h, 0.125*h, 20, 0.05*h)})
	{
		for(int y=1; y<h-1; ++y)
		{
			for(int x=1; x<h-1; ++x)
			{
				bool included=false;
				for(const auto &c:this->circles_)
				{
					if(c.includes(x, y))
					{
						this->buffer_[y][x]=c.v_;
						included=true;
						break;
					}
				}

				if(!included)
				{
					this->mask_.emplace_back(x, y);
				}
			}
		}
	}

	auto solve()
	{
		const auto start=std::chrono::system_clock::now();

		const float min_delta=std::pow(10, -5);
		float max_delta;
		std::vector<float> delta(this->mask_.size());

		std::vector<std::vector<float>> prev;
		int ctr=0;
		
		do
		{
			prev=this->buffer_;
#pragma omp parallel for
			for(int i=0; i<this->mask_.size(); ++i)
			{
				const auto [x, y]=this->mask_[i];
				this->buffer_[y][x]=(prev[y-1][x]+prev[y+1][x]+prev[y][x-1]+prev[y][x+1])/4;
				delta[i]=std::abs(this->buffer_[y][x]-prev[y][x])/100;
			}

			max_delta=*std::max_element(std::execution::par_unseq, delta.begin(), delta.end());

			if(ctr%100==0)
			{
				std::cout<<"loop: "<<ctr<<", delta="<<max_delta<<std::endl;
			}

			++ctr;
		}
		while(max_delta>min_delta);

		return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now()-start).count();
	}

	auto result() const
	{
		return this->buffer_;
	}


private:
	const int h_;
	std::vector<std::vector<float>> buffer_;
	std::vector<std::pair<int, int>> mask_;
	
	class circle
	{
	public:
		circle(const float x, const float y, const int v, const float radius) noexcept
			: x_(x), y_(y), v_(v), radius_(radius)
		{}

		bool includes(const int x, const int y) const noexcept
		{
			return std::pow(x-this->x_, 2)+std::pow(y-this->y_, 2)<=std::pow(this->radius_, 2);
		}

		const float x_, y_;
		const int v_;
		const float radius_;
	};

	std::array<circle, 2> circles_;
};

int main()
{
	constexpr int h=512;
	auto solver=cpu_potential_solver(h);

	const auto time=solver.solve();
	std::cout<<"time: "<<time<<"s"<<std::endl;
	const auto result=solver.result();

	std::ofstream os("./out.csv");
	for(int y=h-1; 0<=y; --y)
	{
		for(int x=0; x<h; ++x)
		{
			os<<result[y][x];

			if(x<h-1)
			{
				os<<",";
			}
		}
		os<<std::endl;
	}

	return 0;
}