/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 *          Luca Heltai, 2021
 */
#include <deal.II/base/utilities.h>

#include "../include/base_problem.h"
#include "../include/linear_elasticity.h"
#include "../include/poisson.h"
#include "../include/stokes.h"


int main(int argc, char **argv) {

	//Decide which program to run, according to the name of the executable
//	const std::string program_name(argv[0]); //the name of the program
//	if (program_name.find("poisson") != std::string::npos) {
//		return run<Poisson<2>>(argc, argv);
//	} else if (program_name.find("linear_elasticity") != std::string::npos) {
//		return run<LinearElasticity<2>>(argc, argv);
//	}else if(program_name.find("stokes") != std::string::npos){
		return run<Stokes<2>>(argc,argv);
//	}
}
