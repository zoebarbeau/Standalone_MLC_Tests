#include <Cabana_Core.hpp>
#include <Cabana_NeighborList.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <iostream>

/*===========================================================
  Execution / Memory Space
===========================================================*/
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace    = ExecutionSpace::memory_space;

/*===========================================================
  Dense Linear Algebra
===========================================================*/
namespace DenseLinearAlgebra
{
template <class Real>
KOKKOS_INLINE_FUNCTION
void matVecMultiply( const Real a[3][3], const Real x[3], Real y[3] )
{
    for ( int i = 0; i < 3; ++i )
    {
        y[i] = 0.0;
        for ( int j = 0; j < 3; ++j )
            y[i] += a[i][j] * x[j];
    }
}
}

/*===========================================================
  Green's Function
===========================================================*/
namespace GreensFunction
{
KOKKOS_INLINE_FUNCTION
void Calculate_qK( const double xp[3], const double xq[3],
                   const double up[3], double K[3],
                   const double h, const int corr_radius )
{
    double dx = xp[0] - xq[0];
    double dy = xp[1] - xq[1];
    double dz = xp[2] - xq[2];

    double r = sqrt(dx*dx + dy*dy + dz*dz);

    double K_M[3][3] =
    {
        { 0.0,  dz,  -dy },
        { -dz, 0.0,  dx },
        { dy,  -dx, 0.0 }
    };

    double delta = sqrt(2.0)*h/2.0;

    if ( r < delta && r > 1e-12 )
    {
        double c =
            1.0/8.0 *
            ( -12.0*(r*r/(delta*delta)) + 20.0 ) /
            (delta*delta*delta) /
            (4.0*Kokkos::numbers::pi);

        for (int i=0;i<3;i++)
            for (int j=0;j<3;j++)
                K_M[i][j] *= c;
    }
    else if ( r >= delta )
    {
        double c = 1.0 / (4.0*Kokkos::numbers::pi*r*r*r);
        for (int i=0;i<3;i++)
            for (int j=0;j<3;j++)
                K_M[i][j] *= c;
    }

    DenseLinearAlgebra::matVecMultiply( K_M, up, K );
}
}

/*===========================================================
  Particle Layout (AoSoA)
===========================================================*/
using ParticleTypes = Cabana::MemberTypes<
    double[3], // position
    double[3], // vorticity
    double[3], // velocity
    double[3]  // vorticity advection
>;

using AoSoA_t = Cabana::AoSoA<ParticleTypes, MemorySpace>;
using HostAoSoA_t = Cabana::AoSoA<ParticleTypes, Kokkos::HostSpace>;
/*===========================================================
  Hill's Vortex Initialization
===========================================================*/
KOKKOS_INLINE_FUNCTION
void init_hills_vortex( const double R,
                        const double U,
                        const double x[3],
                        double vort[3],
                        double vel[3],
			double advect_vort[3] )
{
    double dx = x[0] - 0.5;
    double dy = x[1] - 0.5;
    double dz = x[2] - 0.5;
    double r  = sqrt(dx*dx + dy*dy + dz*dz);

    if ( r < R )
    {
        vort[0] =  15.0*U/(2.0*R*R) * dy;
        vort[1] = -15.0*U/(2.0*R*R) * dx;
        vort[2] =  0.0;
    }
    else
        vort[0] = vort[1] = vort[2] = 0.0;

    advect_vort[0]=advect_vort[1]=advect_vort[2]=0.0;
    vel[0] = vel[1] = vel[2] = 0.0;
}

/*===========================================================
  Neighbor Interaction Kernel
===========================================================*/
template<class NeighborList,class PositionSlice,class AdvectSlice, class VortSlice, class VelSlice>
void run_neighbors( int N,
                    PositionSlice x,
                    VortSlice vort,
                    VelSlice vel,
		    AdvectSlice advectVort,
                    NeighborList& nlist,
                    double h,
                    int corr_radius)
{   std::cout << " N " << N << std::endl;

    using neighbor_traits = Cabana::NeighborList<NeighborList>;
    
    Kokkos::View<std::size_t, MemorySpace> d_total("d_total");
    
    Kokkos::parallel_for(
        "GetTotalNeighbors",
        1,
        KOKKOS_LAMBDA(const int) {
            d_total() = neighbor_traits::totalNeighbor(nlist);
        }
    );
    Kokkos::fence();

    auto h_total =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_total);
    std::cout << "TOTAL NEIGHBORS = " << h_total() << std::endl;   

    Kokkos::Timer timer;  
    auto kernel = KOKKOS_LAMBDA( int p, int q )
         {
             double K[3],Kp[3],Km[3];

             double xp[3] = { x(p,0), x(p,1), x(p,2) };
             double xq[3] = { x(q,0), x(q,1), x(q,2) };
             double uq[3] = { vort(q,0), vort(q,1), vort(q,2) };
             double xplus[3],xminus[3];

	     for(int d = 0; d < 3; d++)
             {
                    xminus[d] = xp[d] - 0.5*h*vort(p,d);
                    xplus[d]  = xp[d] + 0.5*h*vort(p,d);

             }


             GreensFunction::Calculate_qK(
                 xp, xq, uq, K, h, corr_radius );
             GreensFunction::Calculate_qK(
                 xplus, xq, uq, Kp, h, corr_radius );
             GreensFunction::Calculate_qK(
                 xminus, xq, uq, Km, h, corr_radius );

             for (int d=0; d<3; d++){
                 vel(p,d) += K[d];
		 advectVort(p,d) += ( Kp[d] - Km[d] )/ h;
	     }
         };

         Cabana::neighbor_parallel_for(
             Kokkos::RangePolicy<ExecutionSpace>(0,N),
             kernel,
             nlist,
             Cabana::FirstNeighborsTag(),
             Cabana::SerialOpTag(), "neighbor_op"
         );

    Kokkos::fence();
    double time = timer.seconds();
    std::cout << "time= "<< time <<  " Neighbor test complete\n";

}

void generate_hills_vortex(
    int nx,
    double h,
    double R,
    double U,
    std::vector<std::array<double,3>>& pos,
    std::vector<std::array<double,3>>& vort,
    std::vector<std::array<double,3>>& vel
)
{
    for (int i=0;i<nx;i++)
        for (int j=0;j<nx;j++)
            for (int k=0;k<nx;k++)
            {
                double x = i*h; // (i+0.5)*h;
                double y = j*h; // (j+0.5)*h;
                double z = k*h; //  (k+0.5)*h;

                double dx = x - 0.5;
                double dy = y - 0.5;
                double dz = z - 0.5;
                double r  = std::sqrt(dx*dx + dy*dy + dz*dz);

                if (r > R) continue;

                pos.push_back({x,y,z});

                // Hill's vortex vorticity
                vort.push_back({
                    15.0*U/(2.0*R*R) * dy * h*h*h,
                   -15.0*U/(2.0*R*R) * dx * h*h*h,
                    0.0
                });

                vel.push_back({0.0,0.0,0.0});
            }
}

/*===========================================================
  MAIN
===========================================================*/
int main( int argc, char* argv[] )
{
    Kokkos::initialize(argc,argv);
    {
	std::cout << "Kokkos execution space: "
          << ExecutionSpace::name() << std::endl;

        int nx = 64;               // particles per dimension
        int N  = nx*nx*nx;
	int np = 2;
        double h = 1.0 / nx;
	double hp = 1.0/(np*nx);
        int corr_radius = 4;
        double cutoff = 2.5 * h;

        double grid_min[3] = {0.0,0.0,0.0};
        double grid_max[3] = {1.0,1.0,1.0};
        double grid_delta[3] = {h,h,h};
	std::vector<std::array<double,3>> h_pos;
        std::vector<std::array<double,3>> h_vort;
        std::vector<std::array<double,3>> h_vel;
        generate_hills_vortex(nx*np,hp,0.25,1.0,h_pos,h_vort,h_vel );
        int numP = h_pos.size();
        AoSoA_t particles("particles",numP);
        HostAoSoA_t particles_h("particles_h", numP);
        
        auto x_h    = Cabana::slice<0>(particles_h);
        auto vort_h = Cabana::slice<1>(particles_h);
        auto vel_h  = Cabana::slice<2>(particles_h);
        auto advectvort_h =  Cabana::slice<3>(particles_h);

        for (int p = 0; p < numP; ++p){
            for (int d = 0; d < 3; ++d)
            {
                x_h(p,d)    = h_pos[p][d];
                vort_h(p,d) = h_vort[p][d];
                vel_h(p,d)  = h_vel[p][d];
		advectvort_h(p,d) = 0.0;
            }
	}
 
	Cabana::deep_copy(particles, particles_h);

        auto x    = Cabana::slice<0>(particles);
        auto vort = Cabana::slice<1>(particles);
        auto vel  = Cabana::slice<2>(particles);
	auto advectvort = Cabana::slice<3>(particles);

        using ListType = Cabana::LinkedCellList<MemorySpace,double>;      

        //  ListType nlist( x, 0, particles.size(), grid_delta, grid_min, grid_max,corr_radius*h, 0.25 );
	auto nlist = std::make_shared<Cabana::LinkedCellList<MemorySpace,double>>( x, 0, particles.size(), grid_delta, grid_min, grid_max,corr_radius*h, 0.25 );
        run_neighbors(particles.size(),x,vort,vel,advectvort,*nlist,hp,corr_radius);
 
    }
    Kokkos::finalize();
    return 0;
}

