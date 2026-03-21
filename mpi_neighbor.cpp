#include <Cabana_Core.hpp>
#include <Cabana_NeighborList.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <iostream>
#include <array>
#include <mpi.h>
// #include <Cabana_Grid.hpp>
#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_GlobalMesh.hpp>
#include <Cabana_Grid_Partitioner.hpp>
#include <Cabana_Grid_LocalGrid.hpp>
#include <Cabana_Grid_LocalMesh.hpp>
#include <Cabana_Grid_ParticleDistributor.hpp>

/*===========================================================
  Execution / Memory Space
===========================================================*/
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace    = ExecutionSpace::memory_space;

/*===========================================================
  Particle struct for MPI communication
===========================================================*/
struct ParticleMPI{
    double x_send[3];
    double vort_send[3];
    double vel_send[3];
    double advectvort_send[3];
};
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
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc,argv);
    {
        MPI_Comm comm = MPI_COMM_WORLD;
        int rank = 0, nranks = 1;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &nranks);

	    std::cout << "Kokkos execution space: "
          << ExecutionSpace::name() << std::endl;

        int nx = 64;               // particles per dimension
        int N  = nx*nx*nx;
	    int np = 2;
        double h = 1.0 / nx;
	    double hp = 1.0/(np*nx);
        int corr_radius = 4;
        double cutoff = 2.5 * h;
        Cabana::Grid::DimBlockPartitioner<3> dim_block_partitioner;

        // Define the global grid/mesh using Cabana::Grid::createUniformGlobalMesh
        std::array<double,3> grid_min = {0.0,0.0,0.0};
        std::array<double,3> grid_max = {1.0,1.0,1.0};
        std::array<double,3> grid_delta = {h,h,h};
        std::array<int, 3> global_num_cell = {nx,nx,nx};

        std::array<int, 3> ranks_per_dim = dim_block_partitioner.ranksPerDimension(comm, global_num_cell);
        
        auto global_mesh = Cabana::Grid::createUniformGlobalMesh(grid_min, grid_max, grid_delta);
        std::array<bool,3> is_periodic = {true,true,true};
        auto global_grid = Cabana::Grid::createGlobalGrid(comm, global_mesh, is_periodic, dim_block_partitioner);

        // Define the local grid/mesh for each rank using Cabana::Grid::createLocalMesh
        int halo_width = corr_radius; // in terms of number of cells used for defining local grid and applying stencil
        double halo_radius = halo_width * h; // in physical units used for halo exchange
        auto local_grid = Cabana::Grid::createLocalGrid(global_grid, halo_width);
        auto local_mesh = Cabana::Grid::createLocalMesh<MemorySpace>(*local_grid);
        //Print Local and Global mesh information
        auto owned_local = local_grid->indexSpace(Cabana::Grid::Own(),Cabana::Grid::Cell(), Cabana::Grid::Local());
        auto ghost_local = local_grid->indexSpace(Cabana::Grid::Ghost(),Cabana::Grid::Cell(), Cabana::Grid::Local());
        auto owned_global = local_grid->indexSpace(Cabana::Grid::Own(),Cabana::Grid::Cell(), Cabana::Grid::Global());
        // if(rank == 2){
            for (int d = 0; d < 3; ++d){
                std::cout<< "Rank = " << rank << " : " << std::endl;
                std::cout << "  dim " << d << " : [" << owned_local.min(d) << ", " << owned_local.max(d) << ")\n";
                std::cout << "  dim " << d << " : [" << ghost_local.min(d) << ", " << ghost_local.max(d) << ")\n";
                std::cout << "  dim " << d << " : [" << owned_global.min(d) << ", " << owned_global.max(d) << ")\n";
            }
        // --------------------------------------------------
        // Compute owned physical box/subdomain from local grid
        // --------------------------------------------------
        auto mesh = global_grid->globalMesh();

        double x0 = mesh.lowCorner(0);
        double y0 = mesh.lowCorner(1);
        double z0 = mesh.lowCorner(2);

        double dx = mesh.cellSize(0);
        double dy = mesh.cellSize(1);
        double dz = mesh.cellSize(2);

        double xmin = x0 + owned_global.min(0) * dx;
        double xmax = x0 + owned_global.max(0) * dx;
        double ymin = y0 + owned_global.min(1) * dy;
        double ymax = y0 + owned_global.max(1) * dy;
        double zmin = z0 + owned_global.min(2) * dz;
        double zmax = z0 + owned_global.max(2) * dz;
        
        // if ( rank == 0 ){
        //     std::cout << "DimBlockPartitioner ranks_per_dim = {"
        //               << ranks_per_dim[0] << ", "
        //               << ranks_per_dim[1] << ", "
        //               << ranks_per_dim[2] << "}\n";
        //     std::cout << "DimBlockPartitioner rank coordinates = {"
        //               << global_grid->dimBlockId(0) << ", "
        //               << global_grid->dimBlockId(1) << ", "
        //               << global_grid->dimBlockId(2) << "}\n";

        // }
        // --------------------------------------------------
        // 2. Build the 26-neighbor list using blockRank
        // --------------------------------------------------
        // Compute neighboring ranks using ranks per dim and rank coordinates
        int Px = global_grid->dimNumBlock(0);
        int Py = global_grid->dimNumBlock(1);
        int Pz = global_grid->dimNumBlock(2);

        int px = global_grid->dimBlockId(0);
        int py = global_grid->dimBlockId(1);
        int pz = global_grid->dimBlockId(2);

        int bid = global_grid->blockId();

        std::cout << "Rank " << rank
                << " blockId = " << bid
                << " coords = (" << px << "," << py << "," << pz << ")"
                << " process grid = "
                << Px << " x " << Py << " x " << Pz
                << std::endl; 

        // Define a vector to store the neighbor offsets for the 26 neighbors {-1, 0, 1}
        std::vector<std::array<int, 3>> nbr_offsets;
        // Define a vector to store neighbor ranks
        std::vector<int> nbr_ranks;

        //Using blockRank to get neighboring ranks in each direction (with periodicity)
        // Loop over the 3D stencil of neighbors around the current rank's block coordinates (px, py, pz)
        // The stencil includes all combinations of offsets in the x, y, and z directions: -1, 0, and 1
        // THere are 6 faces, 12 edges and 8 corners for a total of 26 neighbors around the current block
        for(int oz=-1; oz<=1; oz++){
            for(int oy=-1; oy<=1; oy++){
                for(int ox=-1; ox<=1; ox++){
                    if(ox==0 && oy==0 && oz==0)
                        continue;

                    int nbr = global_grid->blockRank(px+ox, py+oy, pz+oz);

                    if ( nbr >= 0 ){
                        nbr_offsets.push_back( {ox, oy, oz} );
                        nbr_ranks.push_back( nbr );
                    }
                }
            }
        }

        const std::size_t num_nbrs_ranks = nbr_ranks.size();
        std::cout << "Rank " << rank
                << " has " << num_nbrs_ranks << " neighbors: ";
        for (std::size_t i = 0; i < num_nbrs_ranks; ++i) {
            std::cout << nbr_ranks[i] << " ";
        }
        std::cout << std::endl;
        
        // Creating a deep copy of the neighbor offsets and nbr ranks to the device as they will be used in the lambda for computing the neighbors for halo exchange
        Kokkos::View<int*[3], MemorySpace> device_nbr_offsets("device_nbr_offsets", nbr_offsets.size());
        Kokkos::View<int*, MemorySpace> device_nbr_ranks("device_nbr_ranks", nbr_ranks.size());

        // if(rank == 1){
        //     for(int i = 0; i < nbr_offsets.size(); ++i){
        //         printf("Rank %d neighbor %d: offset = (%d, %d, %d), rank = %d\n", rank, i, nbr_offsets[i][0], nbr_offsets[i][1], nbr_offsets[i][2], nbr_ranks[i]);
        //     }
        // }
        
        // --------------------------------------------------
        // Defining the Hill's vortex initial condition
	    std::vector<std::array<double,3>> h_pos;
        std::vector<std::array<double,3>> h_vort;
        std::vector<std::array<double,3>> h_vel;

        // Generate the Hill's vortex initial condition
        generate_hills_vortex(nx*np,hp,0.25,1.0,h_pos,h_vort,h_vel);
       
    // Figure out how many particles are in the local domain for this rank and create an AoSoA with that many particles. 
    // We will use the same AoSoA for all ranks but only fill the portion of the AoSoA that corresponds to the local domain on each rank. 
    // This is not strictly necessary but it allows us to use the same neighbor list construction and neighbor interaction code on all ranks without having to worry about different data structures on different ranks.
        int numP = h_pos.size();
        std::cout << "nump = " << numP << std::endl;
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

     
        // std::cout << "Rank before migration:" << rank << "Particles size = " << particles.size() << std::endl;

        Cabana::Grid::particleMigrate( *( local_grid ), x, particles, halo_width );

        // slices after migrate as particles may have been reordered during migration
        x    = Cabana::slice<0>(particles);
        vort = Cabana::slice<1>(particles);
        vel  = Cabana::slice<2>(particles);
	    advectvort = Cabana::slice<3>(particles);

        // number of owned particles after migration
        std::size_t num_owned_p = particles.size();
        std::cout << "Rank after migration:" << rank << " Owned particles size = " << num_owned_p << std::endl;

        

        // For particles that are in the halo region and require halo exchange, we need to identify which neighboring ranks they need to be sent to based on their position and the neighbor offsets.
        const int num_nbrs = static_cast<int>(nbr_offsets.size());
        // Creatig a kokkos view for storing the MPI rank/neighbor's offstes for a particle.
        // Does a particle i need to be sent to MPI neighbor nbr_idx? 
        Kokkos::View<int*, MemorySpace> send_flag_nbrs("send_flag_nbrs", num_owned_p);
        Kokkos::View<int*, MemorySpace> offset_scan("offset_scan", num_owned_p);
        Kokkos::View<int, MemorySpace> total_send("total_send");
        // Allocate send and receive buffer based on the total number of particles that need to be sent to this neighbor
        std::vector<Kokkos::View<ParticleMPI*, MemorySpace>> send_buffer(num_nbrs);
        std::vector<Kokkos::View<ParticleMPI*, MemorySpace>> recv_buffer(num_nbrs);
        
        // Send and receive counts for each MPI rank
        std::vector<int> send_counts(num_nbrs, 0);
        std::vector<int> recv_counts(num_nbrs, 0);
        
        // Computing the neighbors for each particle that require halo exchange
        for(int nbr_idx = 0; nbr_idx < num_nbrs; nbr_idx++){
            const int ox = nbr_offsets[nbr_idx][0];
            const int oy = nbr_offsets[nbr_idx][1];
            const int oz = nbr_offsets[nbr_idx][2];

            Kokkos::parallel_for(
            "mark_one_neighbor",
            Kokkos::RangePolicy<ExecutionSpace>(0, num_owned_p),
            KOKKOS_LAMBDA(const int i)
            {
                double xp = x(i,0);
                double yp = x(i,1);
                double zp = x(i,2);

                bool xok = (ox == 0) ||
                           (ox < 0 && xp <  xmin + halo_radius) ||
                           (ox > 0 && xp >= xmax - halo_radius);

                bool yok = (oy == 0) ||
                           (oy < 0 && yp <  ymin + halo_radius) ||
                           (oy > 0 && yp >= ymax - halo_radius);

                bool zok = (oz == 0) ||
                           (oz < 0 && zp <  zmin + halo_radius) ||
                           (oz > 0 && zp >= zmax - halo_radius);

                send_flag_nbrs(i) = (xok && yok && zok) ? 1 : 0;
            });
        
            // Conduct a parallel scan of the flags for this neighbor. This will inform how many particles and which ones need to be sent to the send buffer in what position for this rank.
            Kokkos::parallel_scan(
                "scan_one_neighbor",
                Kokkos::RangePolicy<ExecutionSpace>(0, num_owned_p),
                KOKKOS_LAMBDA(const int i, int& update, const bool final_pass)
            {
                int val = send_flag_nbrs(i);
                if (final_pass){
                    offset_scan(i) = update;
                }
                update += val;

                if (final_pass && i == (int)num_owned_p - 1)
                    total_send() = update;
            });

            int h_total_send = 0;
            Kokkos::deep_copy(h_total_send, total_send);

            // Allocate send buffer based on the total number of particles that need to be sent to this neighbor
            send_buffer[nbr_idx] = Kokkos::View<ParticleMPI*, MemorySpace>("send_buf", h_total_send);
            auto send_buf = send_buffer[nbr_idx];

            // Adding data to send buffer for one MPI rank
            Kokkos::parallel_for(
                "fill_send_buffer_one_neighbor",
                Kokkos::RangePolicy<ExecutionSpace>(0, num_owned_p),
                KOKKOS_LAMBDA(const int i)
                {
                    if (send_flag_nbrs(i) == 1){
                        int os = offset_scan(i);
                        for(int d = 0; d < 3; d++){
                            send_buf(os).x_send[d] = x(i,d);
                            send_buf(os).vort_send[d] = vort(i,d);
                            send_buf(os).vel_send[d] = vel(i,d);
                            send_buf(os).advectvort_send[d] = advectvort(i,d);
                        }
                    }
                }
            );
            // Storing total send in send_count on the host for this neighbor
            send_counts[nbr_idx] = h_total_send;
        }
        Kokkos::fence();

        //---------------------------------------------------------------
        // Exchanging the send and recev counts with the neigbor ranks where the particles need to go from the current rank.
        
        std::vector<MPI_Request> count_reqs(2 * num_nbrs, MPI_REQUEST_NULL); // to track the non-blocking send and receive requests for counts

        for (int nbr_idx = 0; nbr_idx < num_nbrs; ++nbr_idx){
            MPI_Irecv(&recv_counts[nbr_idx], 1, MPI_INT, nbr_ranks[nbr_idx], 100, comm, &count_reqs[2*nbr_idx]);

            MPI_Isend(&send_counts[nbr_idx], 1, MPI_INT, nbr_ranks[nbr_idx], 100, comm, &count_reqs[2*nbr_idx + 1]);
        }

        MPI_Waitall(2 * num_nbrs, count_reqs.data(), MPI_STATUSES_IGNORE);

        // ---------------------------------------------------------------
        // Allocating recv buffer and posting receive data for each rank based on the recv counts

        std::vector<MPI_Request> data_reqs(2 * num_nbrs, MPI_REQUEST_NULL); //to track the non-blocking send and receive requests for data
        
        for(int nbr_idx = 0; nbr_idx < num_nbrs; ++nbr_idx){
            recv_buffer[nbr_idx] = Kokkos::View<ParticleMPI*, MemorySpace>("recv_buf", recv_counts[nbr_idx]);

            //MPI_Irecv
            MPI_Irecv(recv_buffer[nbr_idx].data(), recv_counts[nbr_idx] * sizeof(ParticleMPI), MPI_BYTE, nbr_ranks[nbr_idx], 200, comm, &data_reqs[2*nbr_idx]);

            // MPI_Isend
            MPI_Isend(send_buffer[nbr_idx].data(), send_counts[nbr_idx] * sizeof(ParticleMPI), MPI_BYTE, nbr_ranks[nbr_idx], 200, comm, &data_reqs[2*nbr_idx + 1]);
        }  

        MPI_Waitall(2 * num_nbrs, data_reqs.data(), MPI_STATUSES_IGNORE);

    // ---------------------------------------------------------------
    // Creating the linked cell list for the  owned + ghost particles after migration and halo exchange.
    // ---------------------------------------------------------------

    // Storing the new number of particles after halo exchange.
    int num_ghosts_global = 0;
    for (int nbr_idx = 0; nbr_idx < num_nbrs; ++nbr_idx){
        num_ghosts_global += recv_counts[nbr_idx];
    }
    std::size_t total_particles_after_halo = num_owned_p + num_ghosts_global;
    // New AoSoA with owned + ghost particle data
    AoSoA_t particles_w_ghost("particles_w_ghost",total_particles_after_halo);

    auto x_wg    = Cabana::slice<0>(particles_w_ghost);
    auto vort_wg = Cabana::slice<1>(particles_w_ghost);
    auto vel_wg  = Cabana::slice<2>(particles_w_ghost);
	auto advectvort_wg = Cabana::slice<3>(particles_w_ghost);

    Kokkos::parallel_for(
        "fill_particles_w_ghost",
        Kokkos::RangePolicy<ExecutionSpace>(0, num_owned_p),
        KOKKOS_LAMBDA(const int i)
        {
                for (int d = 0; d < 3; ++d){
                    x_wg(i,d) = x(i,d);
                    vort_wg(i,d) = vort(i,d);
                    vel_wg(i,d) = vel(i,d);
                    advectvort_wg(i,d) = advectvort(i,d);
                }
        });
    // Filling the ghost particle data in the new AoSoA based on the data received from the neighbors
    std::size_t ghost_base = num_owned_p; // base index for ghost particles in the new AoSoA
    for (int nbr_idx = 0; nbr_idx < num_nbrs; ++nbr_idx){
        auto recv_buf = recv_buffer[nbr_idx];
        const int nrecv = recv_counts[nbr_idx];
        const std::size_t base = ghost_base;

        Kokkos::parallel_for(
            "append_ghost_particles",
            Kokkos::RangePolicy<ExecutionSpace>(0, nrecv),
            KOKKOS_LAMBDA(const int j){
                const std::size_t p = base + j;

                for (int d = 0; d < 3; ++d){
                    x_wg(p,d)    = recv_buf(j).x_send[d];
                    vort_wg(p,d) = recv_buf(j).vort_send[d];
                    vel_wg(p,d)  = recv_buf(j).vel_send[d];
                    advectvort_wg(p,d)  = recv_buf(j).advectvort_send[d];
                }
            }
        );
        ghost_base += nrecv;
    }
    Kokkos::fence();

    particles = particles_w_ghost; // replace the original AoSoA with the new one that has ghost particles

    x = Cabana::slice<0>(particles);
    vort = Cabana::slice<1>(particles);
    vel = Cabana::slice<2>(particles);
    advectvort = Cabana::slice<3>(particles);

    // Defining the linked cell list
    // Defining the new local min and max for the linked cell list based on woed + ghost particles

    std::array<double, 3> local_grid_min = {xmin - halo_radius, ymin - halo_radius, zmin - halo_radius};
    std::array<double, 3> local_grid_max = {xmax + halo_radius, ymax + halo_radius, zmax + halo_radius};

    using ListType = Cabana::LinkedCellList<MemorySpace,double>;
    auto nlist = std::make_shared<Cabana::LinkedCellList<MemorySpace,double>>( x, 0, particles.size(), grid_delta, local_grid_min, local_grid_max,halo_radius, 0.25 );

    // Computing local particle interations
    run_neighbors(particles.size(),x,vort,vel,advectvort,*nlist,hp,corr_radius);
 
}
    Kokkos::finalize();
    return 0;
}

