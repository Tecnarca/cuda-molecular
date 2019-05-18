#ifndef CHECK_KERNEL_HEAD
#define CHECK_KERNEL_HEAD
#include <check_kerns.h>
#include <array>
#include <cmath>
#include <map>
#include <vector>
static constexpr double pi = 3.141592653589793238462643383279;
static constexpr double radian_coef = pi / 128.0;
static constexpr double grid_factor_d = 0.5;


	struct rigid_rotation
	{
		using value_type = std::array<float, 9>;

		// the angles are expressed in degree
		static inline value_type compute_matrix( const float alpha, const float beta )
		{
			// precompute the requested values
			const float alpha_rad = alpha * radian_coef;
			const float beta_rad = beta * radian_coef;
			const float cosalpha = cos(alpha_rad);
			const float sinalpha = sin(alpha_rad);
			const float cosbeta = cos(beta_rad);
			const float sinbeta = sin(beta_rad);

			// return the actual matrix
			return {{
					cosalpha * cosbeta,
					sinalpha * cosbeta,
					-sinbeta,

					-sinalpha,
					cosalpha,
					0.0,

					cosalpha * sinbeta,
					sinalpha * sinbeta,
					cosbeta
				}};
		}
	};

    // compute a rotation matrix wrt an arbitrary axis
    struct free_rotation
    {
        using value_type = std::array<float, 12>;

        // the angles are expressed in degree
        static inline value_type compute_matrix( const float rotation_angle,
                const float x_orig, const float y_orig, const float z_orig,
                const float x_vector, const float y_vector, const float z_vector )
        {
            // compute the rotation axis
            const float u = static_cast<float>(x_vector - x_orig);
            const float v = static_cast<float>(y_vector - y_orig);
            const float w = static_cast<float>(z_vector - z_orig);
            const float u2 = u * u;
            const float v2 = v * v;
            const float w2 = w * w;

            // compute its lenght and square root
            const float l2 = u * u + v * v + w * w;
            const float l = sqrt(l2);

            // precompute sine and cosine for the angle
            const float angle_rad = radian_coef * -rotation_angle;
            const float sint = sin(angle_rad);
            const float cost = cos(angle_rad);
            const float one_minus_cost = static_cast<float>(1) - cost;

            // return the actual matrix
            return {{
                    (u2 + (v2 + w2) * cost) / l2,
                    (u* v * one_minus_cost - w* l * sint) / l2,
                    (u* w * one_minus_cost + v* l * sint) / l2,
                    ((x_orig * (v2 + w2) - u * (y_orig * v + z_orig * w)) * one_minus_cost + (y_orig * w - z_orig * v) * l * sint) / l2,

                    (u* v * one_minus_cost + w* l * sint) / l2,
                    (v2 + (u2 + w2) * cost) / l2,
                    (v* w * one_minus_cost - u* l * sint) / l2,
                    ((y_orig * (u2 + w2) - v * (x_orig * u + z_orig * w)) * one_minus_cost + (z_orig * u - x_orig * w) * l * sint) / l2,

                    (u* w * one_minus_cost - v* l * sint) / l2,
                    (v* w * one_minus_cost + u* l * sint) / l2,
                    (w2 + (u2 + v2) * cost) / l2,
                    ((z_orig * (u2 + v2) - w * (x_orig * u + y_orig * v)) * one_minus_cost + (x_orig * v - y_orig * u) * l * sint) / l2
                }};
        }
    };


	template <int n_atoms>
	inline void rotate( float* atoms, const rigid_rotation::value_type& rotation_matrix )
	{
		// get some constants out of the problem
		const float m11 = static_cast<float>(std::get<0>(rotation_matrix));
		const float m12 = static_cast<float>(std::get<1>(rotation_matrix));
		const float m13 = static_cast<float>(std::get<2>(rotation_matrix));
		const float m21 = static_cast<float>(std::get<3>(rotation_matrix));
		const float m22 = static_cast<float>(std::get<4>(rotation_matrix));
		const float m23 = static_cast<float>(std::get<5>(rotation_matrix));
		const float m31 = static_cast<float>(std::get<6>(rotation_matrix));
		const float m32 = static_cast<float>(std::get<7>(rotation_matrix));
		const float m33 = static_cast<float>(std::get<8>(rotation_matrix));

		// rotate the ligand atoms
		for ( int i= 0; i < n_atoms; ++i )
		{
			// take the previous value of the atoms
			const float prev_x = atoms[i];
			const float prev_y = atoms[i+n_atoms];
			const float prev_z = atoms[i+2*n_atoms];

			// update the position
			atoms[i] 	   = m11 * prev_x + m12 * prev_y + m13 * prev_z;
			atoms[i+n_atoms]   = m21 * prev_x + m22 * prev_y + m23 * prev_z;
			atoms[i+2*n_atoms] = m31 * prev_x + m32 * prev_y + m33 * prev_z;
		}
	}

	template <int n_atoms>
    inline void rotate( float* in,
                        int* mask,
                        const free_rotation::value_type& rotation_matrix)
    {
        // get some constant out of the problem
        const float m11 = static_cast<float>(std::get<0>(rotation_matrix));
        const float m12 = static_cast<float>(std::get<1>(rotation_matrix));
        const float m13 = static_cast<float>(std::get<2>(rotation_matrix));
        const float m14 = static_cast<float>(std::get<3>(rotation_matrix));
        const float m21 = static_cast<float>(std::get<4>(rotation_matrix));
        const float m22 = static_cast<float>(std::get<5>(rotation_matrix));
        const float m23 = static_cast<float>(std::get<6>(rotation_matrix));
        const float m24 = static_cast<float>(std::get<7>(rotation_matrix));
        const float m31 = static_cast<float>(std::get<8>(rotation_matrix));
        const float m32 = static_cast<float>(std::get<9>(rotation_matrix));
        const float m33 = static_cast<float>(std::get<10>(rotation_matrix));
        const float m34 = static_cast<float>(std::get<11>(rotation_matrix));

        // process it as fast as you can
        for (int i = 0; i < n_atoms; ++i)
        {
            // make sure to consider only the fragment atoms
            if (mask[i] == 1)
            {
                // take the previous value of the atoms
                const float prev_x = in[i] 	    ;
                const float prev_y = in[i+n_atoms]  ;
                const float prev_z = in[i+2*n_atoms];

                // compute the next values
                in[i] 	   	= m11 * prev_x + m12 * prev_y + m13 * prev_z + m14;
                in[i+n_atoms]   = m21 * prev_x + m22 * prev_y + m23 * prev_z + m24;
                in[i+2*n_atoms] = m31 * prev_x + m32 * prev_y + m33 * prev_z + m34;
            }
        }
    }

	template <int n_atoms>
	int measure_shotgun (float* atoms, float* pocket)
	{

		// get the average score
		int score = static_cast<int>(0);

		// loop over the atoms and get the pocket score
		for ( int i =0; i < n_atoms; ++i )
		{
			// compute the index inside of the pocket
			int index_x = static_cast<int>(atoms[i]  * grid_factor_d );
			int index_y = static_cast<int>(atoms[i+n_atoms]  * grid_factor_d );
			int index_z = static_cast<int>(atoms[i+2*n_atoms]  * grid_factor_d );
			if (index_x < 0) index_x = 0;
			if (index_x > 100) index_x = 100;
			if (index_y < 0) index_y = 0;
			if (index_y > 100) index_y = 100;
			if (index_z < 0) index_z = 0;
			if (index_z > 100) index_z = 100;
			// update the score value
			score += pocket[index_x+100*index_y+10000*index_z];
		}
		return score;
	}

	template <int n_atoms>
	void align_check( float* in, float* out,int precision, float* scores)
	{

		// then we generate the rigid rotation matrices
		const auto step_x = rigid_rotation::compute_matrix(precision,0);
		const auto step_y = rigid_rotation::compute_matrix(0,precision);
		float* backup = new float [n_atoms*3];
		std::map<int, std::map<int,int> > dbg_helper;


		for(int i=0;i<n_atoms;i++)
		{
			backup[i] 	   =	  in[i] 	  ;
						backup[i+n_atoms]  =	  in[i+n_atoms]   ;
						backup[i+2*n_atoms]=	  in[i+2*n_atoms] ;
		}
		// get the initial evaluation
		auto best_score = measure_shotgun<n_atoms>(in, scores);
		int best_angle_x = 0;
		int best_angle_y = 0;

		// loop over the first axis
		for ( int i = 0/* precision*/; i </*=*/ static_cast<int>(256); i += precision )
		{
			dbg_helper.insert(std::make_pair (i,std::map<int,int>()));
			// loop over the second axis
			for ( int j = 0; j < static_cast<int>(256); j += precision )
			{

				// measure the pacman score
				const auto actual_score = measure_shotgun<n_atoms>(in, scores);
				dbg_helper[i].insert(std::make_pair(j,actual_score));
					//std::cout<<"score (cpu): "<<actual_score<<", alfa angle is: "<< i <<" and beta angle is: "<<j<<std::endl;
				// update the scoring function
				if (actual_score > best_score)
				{
					best_score = actual_score;
					best_angle_x = i;
					best_angle_y = j;
				}
				rotate<n_atoms>(in, step_y);
			}
			rotate<n_atoms>(in, step_x);

		}
		//DEBUG
		/*
		for (int i=0;i<256; i+=precision)
			for (int j=0;j<256; j+=precision)
				std::cout<<"score is: "<<dbg_helper.at(j).at(i)<<" for angle x: "<<j<<" and y: "<<i<<std::endl;


		std::cout<<" END: best angle is: "<<best_angle_x<<" and y: "<<best_angle_y<<std::endl;
*/
//		std::cout<<"xpos is: "<<in[0]<<std::endl;
		// place the ligand in the best position found
		const auto optimal_rotation = rigid_rotation::compute_matrix(static_cast<float>(best_angle_x), static_cast<float>(best_angle_y));
		rotate<n_atoms>(backup, optimal_rotation);
		for(int i=0; i<n_atoms; i++)
		{
			out[i]	 	=  backup[i];
///			std::cout<<"out "<< i <<" is: "<<out[i]<<std::endl;
			out[i+n_atoms]  =  backup[i+n_atoms]  ;
			out[i+2*n_atoms]=  backup[i+2*n_atoms];
		}
	}



	template<int n_atoms>
    inline bool fragment_is_bumping( const float* in, const int* mask)
    {
        // set the bumping threshold
        const float limit_distance2 = 2.0f; 

        // this is the variable of the result
        bool is_bumping = false;

        // loop over the atoms of the ligand
        for ( int i = 0; i < n_atoms; ++i )
        {
            // perform the triangular comparison
            for ( int j = i +1; j < n_atoms; ++j )
            {
                const bool evaluate_fragment = std::abs(mask[i] - mask[j]) == 1;

                // we need to check only bumps between fragment and the remainder of the molecule
                if (evaluate_fragment)
                {
                    const float diff_x = in[i] - in[j];
                    const float diff_y = in[i+n_atoms] - in[j+n_atoms];
                    const float diff_z = in[i+2*n_atoms] - in[j+2*n_atoms];
                    const float distance2 = diff_x * diff_x +  diff_y * diff_y +  diff_z * diff_z;

                    // update step
                    is_bumping = is_bumping || distance2 < limit_distance2;
			if (is_bumping)
			{
//			std::cout<<"bumps"<<std::endl;
			return true;
			}
                }
            }
        }

        // if we reach this point, the ligand is not bumping
        return is_bumping;
    }




	// this function rotates the fragment to maximize the pacman score
	template<int n_atoms, int n_frags>
	void ps_check(float* in, float* out, int precision,float* score_pos, int* start, int* stop, int* mask )
	{
		// optimize each rotamer
		for ( int i = 0; i < n_frags; ++i )
		{
        		// compute the epsilon value for floating point precision
		        const auto epsilon = std::numeric_limits<float>::epsilon();

		        // get the index of starting atom
		        const auto start_atom_index = start[i];
		        const auto stop_atom_index = stop[i];
			
			// get the rotation matrix for this fragment
			const auto rotation_matrix = free_rotation::compute_matrix(precision,in[start_atom_index],in[start_atom_index+n_atoms],in[start_atom_index+2*n_atoms],in[stop_atom_index],in[stop_atom_index+n_atoms],  in[stop_atom_index+2*n_atoms]);
			// declare the variables for driving the optimization
			int best_angle = 0;
			int best_score = measure_shotgun<n_atoms>(in, score_pos);
			    std::cout<<"init: best score is: "<<best_score<<std::endl;
			bool is_best_bumping = fragment_is_bumping<n_atoms>(in, &mask[i*n_atoms]);
			// optimize shape
			for ( int j = 0 ; j < 256; j += precision )
			{
			// rotate the fragment
#if 0
			if (i==0 && j==32)
			{
				for (int printi = 0; printi<3;printi++)
				{
					for (int printk=0; printk<64;printk++)
					std::cout<< in[64*printi+printk]<<'\t';
				std::cout<<std::endl<<std::endl;
				}
			}
#endif
			// compute the score
			    const int score = measure_shotgun<n_atoms>(in, score_pos);
			    std::cout<<" score is: "<<score<<" for fragm: "<<i<<"with angle: "<<j<<std::endl;
			    // check if we have to update the best
			    if (score > best_score)
			    {
				const bool is_bumping = fragment_is_bumping<n_atoms>(in, &mask[i*n_atoms]);

				if ((is_bumping == is_best_bumping) || ((!is_bumping) && is_best_bumping))
				{
				    best_score = score;
				    best_angle = j;
				}
			    }
			    else // the actual angle is not the best one
			    {
				if (is_best_bumping)
				    {
					   const bool is_bumping = fragment_is_bumping<n_atoms>(in, &mask[i*n_atoms]);
		
					   if (!is_bumping)
					   {
						best_score = score;
						best_angle = j;
						is_best_bumping = false;
					    }
				    }
			    }
			    rotate<n_atoms>(in, &mask[i*n_atoms], rotation_matrix);
			}
			std::cout<<"best angle is: "<<best_angle<<std::endl;
			const int score = measure_shotgun<n_atoms>(in, score_pos);
			std::cout<<" score is: "<<score<<" for fragm: "<<i<<"with angle out"<<std::endl;
			const auto rotation_matrix_best = free_rotation::compute_matrix(best_angle,in[start_atom_index],in[start_atom_index+n_atoms],in[start_atom_index+2*n_atoms],in[stop_atom_index],in[stop_atom_index+n_atoms],  in[stop_atom_index+2*n_atoms]);
    			rotate<n_atoms>(in, &mask[i*n_atoms], rotation_matrix_best );
    		}

		for (int i=0;i < n_atoms*3;i++)
			out[i] = in [i];
	}












#endif
