#include "../include/stokes.h"

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/trilinos_linear_operator.h>



#include <deal.II/lac/solver_gmres.h>

#include <deal.II/numerics/error_estimator.h>



#include <utility>

using namespace dealii;

namespace
{
  std::vector<std::string>
  get_component_names(int dim)
  {
    std::vector<std::string> names(dim + 1, "u");
    names[dim] = "p";
    return names;
  }
} // namespace

template <int dim>
Stokes<dim>::Stokes()
  : BaseBlockProblem<dim>(get_component_names(dim),
                          "Stokes<" + std::to_string(dim) + ">")
//  , velocity(0)
//  , pressure(dim)
{
  // Output the vector result.
  this->add_data_vector.connect([&](auto &data_out) {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim + 1,
                     DataComponentInterpretation::component_is_part_of_vector);

    interpretation[dim] = DataComponentInterpretation::component_is_scalar;

    data_out.add_data_vector(this->locally_relevant_block_solution,
                             this->component_names,
                             DataOut<dim>::type_dof_data,
                             interpretation);
  });
}



template <int dim>
void
Stokes<dim>::assemble_system_one_cell(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData &                                         scratch,
  CopyData &                                            copy)
{
  auto &cell_matrix = copy.matrices[0];
  auto &cell_matrix2 = copy.matrices[1];
  auto &cell_rhs    = copy.vectors[0];

  cell->get_dof_indices(copy.local_dof_indices[0]);

  const auto &fe_values = scratch.reinit(cell);
  cell_matrix           = 0;
  cell_matrix2 			= 0;
  cell_rhs              = 0;

  for (const unsigned int q_index : fe_values.quadrature_point_indices())
    {
      for (const unsigned int i : fe_values.dof_indices())
        {
          const auto eps_v = fe_values[BaseBlockProblem<dim>::velocity].symmetric_gradient(
            i, q_index); // SymmetricTensor<2,dim>
          const auto div_v =
            fe_values[BaseBlockProblem<dim>::velocity].divergence(i, q_index);         // double
          const auto q = fe_values[BaseBlockProblem<dim>::pressure].value(i, q_index); // double

          for (const unsigned int j : fe_values.dof_indices())
            {
              const auto eps_u = fe_values[BaseBlockProblem<dim>::velocity].symmetric_gradient(
                j, q_index); // SymmetricTensor<2,dim>
              const auto div_u =
                fe_values[BaseBlockProblem<dim>::velocity].divergence(j, q_index);         // double
              const auto p = fe_values[BaseBlockProblem<dim>::pressure].value(j, q_index); // double

              cell_matrix(i, j) +=
                (2*scalar_product(eps_v, eps_u) - p * div_v - q * div_u) *
                fe_values.JxW(q_index); // dx

              cell_matrix2(i,j)+= q*p * fe_values.JxW(q_index);
            }

        }
      for (const unsigned int i : fe_values.dof_indices())
        {
          const auto comp_i = this->fe->system_to_component_index(i).first;
          cell_rhs(i) +=
            (fe_values.shape_value(i, q_index) * // phi_i(x_q)
             this->forcing_term.value(fe_values.quadrature_point(q_index),
                                      comp_i) * // f(x_q)
             fe_values.JxW(q_index));           // dx
        }
    }

  if (cell->at_boundary())
    //  for(const auto face: cell->face_indices())
    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      if (this->neumann_ids.find(cell->face(f)->boundary_id()) !=
          this->neumann_ids.end())
        {
          auto &fe_face_values = scratch.reinit(cell, f);
          for (const unsigned int q_index :
               fe_face_values.quadrature_point_indices())
            for (const unsigned int i : fe_face_values.dof_indices())
              {
                const auto comp_i =
                  this->fe->system_to_component_index(i).first;
                cell_rhs(i) +=
                  fe_face_values.shape_value(i, q_index) *
                  this->neumann_boundary_condition.value(
                    fe_face_values.quadrature_point(q_index), comp_i) *
                  fe_face_values.JxW(q_index);//dx
              }
        }





}


template <int dim>
void
Stokes<dim>::solve()
{

	TimerOutput::Scope                timer_section(this->timer, "solve");

    SolverControl solver_control_inner(5000, 1e-5);
    SolverCG<LA::MPI::Vector> solver_CG(solver_control_inner);


    LA::MPI::PreconditionAMG prec_A;
    prec_A.initialize(this->system_block_matrix.block(0, 0));




    LA::MPI::PreconditionAMG prec_S;
    prec_S.initialize(this->preconditioner_matrix.block(1, 1));

    const auto A = linear_operator<LA::MPI::Vector>(this->system_block_matrix.block(0, 0));
    auto Bt = linear_operator< LA::MPI::Vector >( this->system_block_matrix.block(0,1) );

    auto ZeroP = linear_operator< LA::MPI::Vector >( this->system_block_matrix.block(1,1) );



    const auto amgA = linear_operator(A, prec_A);

    const auto S =
      linear_operator<LA::MPI::Vector>(this->preconditioner_matrix.block(1, 1));
    const auto amgS = linear_operator(S, prec_S);
    ReductionControl          inner_solver_control(100,
                                          1e-8 * this->system_rhs.l2_norm(),
                                          1.e-2);
    SolverCG<LA::MPI::Vector> cg(inner_solver_control);
    const auto minus_invS = inverse_operator(-1*S, cg, amgS);


    const auto Mat = block_operator<2, 2, LA::MPI::BlockVector >({
              {
                {{ A, Bt }} ,
                {{ ZeroP, -1*S }}
              }
            } );

    const auto Ainv = inverse_operator(A, cg,amgA);
    const auto diagInv = block_diagonal_operator<2, LA::MPI::BlockVector>(std::array<::LinearOperator<typename LA::MPI::BlockVector::BlockType>,2>{{Ainv, minus_invS}});



    const auto P_inv = block_back_substitution<LA::MPI::BlockVector>(Mat, diagInv);




    SolverControl solver_control(this->system_block_matrix.m(), 1e-10 * this->system_block_rhs.l2_norm());


    SolverFGMRES<LA::MPI::BlockVector> solver(solver_control);

    solver.solve(this->system_block_matrix,
        		this->block_solution,
    			this->system_block_rhs,P_inv);


    this->constraints.distribute(this->block_solution);
    this->locally_relevant_block_solution = this->block_solution;


}



template <int dim>
void
Stokes<dim>::estimate()
{
  TimerOutput::Scope timer_section(this->timer, "estimate");
  if (this->estimator_type == "kelly")
    {
      std::map<types::boundary_id, const Function<dim> *> neumann;
      for (const auto id : this->neumann_ids)
        neumann[id] = &this->neumann_boundary_condition;

      QGauss<dim - 1> face_quad(this->fe->degree + 1);
      KellyErrorEstimator<dim>::estimate(*this->mapping,
                                         this->dof_handler,
                                         face_quad,
                                         neumann,
                                         this->locally_relevant_block_solution,
                                         this->error_per_cell,
                                         this->fe->component_mask(BaseBlockProblem<dim>::velocity));
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }
  auto global_estimator = this->error_per_cell.l2_norm();
  this->error_table.add_extra_column("estimator", [global_estimator]() {
    return global_estimator;
  });
  this->error_table.error_from_exact(*this->mapping,
                                     this->dof_handler,
                                     this->locally_relevant_block_solution,
                                     this->exact_solution);
}



template class Stokes<1>;
template class Stokes<2>;
template class Stokes<3>;
