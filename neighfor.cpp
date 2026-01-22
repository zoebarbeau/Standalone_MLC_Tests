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
    double[3]  // velocity
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
                        double vel[3] )
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

    vel[0] = vel[1] = vel[2] = 0.0;
}

/*===========================================================
  Uniform Particle Positions
===========================================================*/
template<class PositionSlice>
void create_positions( PositionSlice x, int nx )
{
    int N = nx*nx*nx;

    Kokkos::parallel_for(
        "init_positions",
        Kokkos::RangePolicy<ExecutionSpace>(0,N),
        KOKKOS_LAMBDA(int p)
    {
        int i = p % nx;
        int j = (p/nx) % nx;
        int k = p/(nx*nx);

        x(p,0) = (i+0.5)/nx;
        x(p,1) = (j+0.5)/nx;
        x(p,2) = (k+0.5)/nx;
    });
}

/*===========================================================
  Neighbor Interaction Kernel
===========================================================*/
template<class NeighborList,class PositionSlice, class VortSlice, class VelSlice>
void run_neighbors( int N,
                    PositionSlice x,
                    VortSlice vort,
                    VelSlice vel,
                    NeighborList& nlist,
                    double h,
                    int corr_radius,
		    bool use_outer)
{   std::cout << " N " << N << std::endl;

    using neighbor_traits = Cabana::NeighborList<NeighborList>;
    Kokkos::Timer timer;  
    if( use_outer ) {
        Kokkos::View<int*, MemorySpace> d_counts("d_counts", N);
        Kokkos::View<int*, Kokkos::HostSpace> h_counts("h_counts", N);
    
        // Fill counts on device
        Kokkos::parallel_for("ComputeNeighborCounts", N, KOKKOS_LAMBDA(int q){
            d_counts(q) = neighbor_traits::numNeighbor(nlist, q);
        });
        Kokkos::fence();
    
        // Copy to host
        Kokkos::deep_copy(h_counts, d_counts);
    
        // Serial loop over q, parallel over neighbors
        for(int q=0; q<N; ++q){
            int numNeigh = h_counts(q);
            Kokkos::parallel_for(
                "interactions2",
                Kokkos::RangePolicy<ExecutionSpace>(0,numNeigh),
                KOKKOS_LAMBDA(int i){
                    int p = neighbor_traits::getNeighbor(nlist, q, i);
    
                    // rest of your kernel
                }
            );
//            Kokkos::fence();
        }
    
    }else{
         auto kernel = KOKKOS_LAMBDA( int p, int q )
         {
             double K[3];

             double xp[3] = { x(p,0), x(p,1), x(p,2) };
             double xq[3] = { x(q,0), x(q,1), x(q,2) };
             double uq[3] = { vort(q,0), vort(q,1), vort(q,2) };

             GreensFunction::Calculate_qK(
                 xp, xq, uq, K, h, corr_radius );

             for (int d=0; d<3; d++)
                 vel(p,d) += K[d];
         };

         Cabana::neighbor_parallel_for(
             Kokkos::RangePolicy<ExecutionSpace>(0,N),
             kernel,
             nlist,
             Cabana::FirstNeighborsTag(),
             Cabana::SerialOpTag(), "neighbor_op"
         );
    }

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
                double x = (i+0.5)*h;
                double y = (j+0.5)*h;
                double z = (k+0.5)*h;

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

        int nx = 128;                // particles per dimension
        int N  = nx*nx*nx;
	int np = 2;
        double h = 1.0 / nx;
	double hp = 1.0/(np*nx);
        int corr_radius = 4;
        double cutoff = 2.5 * h;

        bool use_verlet = true;      // switch neighbor list
        bool use_outer  = false;
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
        
        for (int p = 0; p < numP; ++p){
            for (int d = 0; d < 3; ++d)
            {
                x_h(p,d)    = h_pos[p][d];
                vort_h(p,d) = h_vort[p][d];
                vel_h(p,d)  = h_vel[p][d];
            }
	}
 
	Cabana::deep_copy(particles, particles_h);


        auto x    = Cabana::slice<0>(particles);
        auto vort = Cabana::slice<1>(particles);
        auto vel  = Cabana::slice<2>(particles);

        if (use_verlet)
        {   

           using ListAlgorithm = Cabana::FullNeighborTag;
           using ListType =
                 Cabana::VerletList<MemorySpace, ListAlgorithm, Cabana::VerletLayoutCSR,
                           Cabana::TeamOpTag>;	
	
       
          double cutoff = corr_radius * h;
          double skin   = h / cutoff;
       
          ListType verlet_list(
           x,
           0,
           numP,
           cutoff,
           skin,
           grid_min,
           grid_max
          );
       
           
           // run neighbors
           run_neighbors(numP, x, vort, vel, verlet_list, hp, corr_radius,use_outer);

        }
        else
        {
            using ListType = Cabana::LinkedCellList<MemorySpace,double>;        
            ListType nlist( x, 0, particles.size(), grid_delta, grid_min, grid_max,corr_radius*h, 0.25 );
            run_neighbors(particles.size(),x,vort,vel,nlist,hp,corr_radius, use_outer);
        }
 
    }
    Kokkos::finalize();
    return 0;
}

