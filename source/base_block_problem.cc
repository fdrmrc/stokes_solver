#include "../include/base_block_problem.h"

#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_tools.h>

#include <deal.II/lac/linear_operator.h>



using namespace dealii;





//
template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues()
    : Function<dim>(dim + 1)
  {}
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;
  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &  value) const override;
};


template <int dim>
double BoundaryValues<dim>::value(const Point<dim> & p,
                                  const unsigned int component) const
{
  Assert(component < this->n_components,
         ExcIndexRange(component, 0, this->n_components));
  if (std::abs(p[1]-1.0) < 1e-14 && component == 0){
	  return 1.0;
  }
  return 0;
}

template <int dim>
void BoundaryValues<dim>::vector_value(const Point<dim> &p,
                                       Vector<double> &  values) const
{
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = BoundaryValues<dim>::value(p, c);
}






//
//
//template <int dim>
//class BoundaryValues : public Function<dim>
//{
//public:
//  BoundaryValues()
//    : Function<dim>(dim + 1)
//  {}
//  virtual double value(const Point<dim> & p,
//                       const unsigned int component) const override;
//};
//template <int dim>
//double BoundaryValues<dim>::value(const Point<dim> & p,
//                                  const unsigned int component) const
//{
//  Assert(component < this->n_components,
//         ExcIndexRange(component, 0, this->n_components));
//  if (component == 0 && std::abs(p[dim - 1] - 1.0) < 1e-10)
//    return 1.0;
//  return 0;
//}
//








template <int dim>
BaseBlockProblem<dim>::BaseBlockProblem(
  const std::vector<std::string> component_names,
  const std::string &            problem_name)
  : BaseProblem<dim>(component_names.size(), problem_name)
  , component_names(component_names)
  ,velocity(0)
  ,pressure(dim)
{}



template <int dim>
void
BaseBlockProblem<dim>::setup_system()
{
  TimerOutput::Scope timer_section(this->timer, "setup_system");
  if (!this->fe)
    {
      this->fe = FETools::get_fe_by_name<dim>(this->fe_name);
      this->mapping =
        std::make_unique<MappingQGeneric<dim>>(this->mapping_degree);
      const auto vars = dim == 1 ? "x" : dim == 2 ? "x,y" : "x,y,z";
      this->forcing_term.initialize(vars,
                                    this->forcing_term_expression,
                                    this->constants);
      this->exact_solution.initialize(vars,
                                      this->exact_solution_expression,
                                      this->constants);

      this->dirichlet_boundary_condition.initialize(
        vars, this->dirichlet_boundary_conditions_expression, this->constants);

      this->neumann_boundary_condition.initialize(
        vars, this->neumann_boundary_conditions_expression, this->constants);
    }

  this->dof_handler.distribute_dofs(*this->fe);

  // renumber dofs in a blockwise manner.
  std::vector<unsigned int> blocks(this->n_components);
  unsigned int              i = 0;
  Assert(component_names.size() > 0, ExcInternalError());
  AssertDimension(this->n_components, component_names.size());
  blocks[0] = i;
  for (unsigned int j = 1; j < this->n_components; ++j)
    {
      if (component_names[j] == component_names[j - 1])
        blocks[j] = i;
      else
        blocks[j] = ++i;
    }
  DoFRenumbering::component_wise(this->dof_handler, blocks);

  dofs_per_block = DoFTools::count_dofs_per_fe_block(this->dof_handler, blocks);

  locally_owned_dofs =
    this->dof_handler.locally_owned_dofs().split_by_block(dofs_per_block);


  IndexSet non_blocked_locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(this->dof_handler,
                                          non_blocked_locally_relevant_dofs);
  locally_relevant_dofs =
    non_blocked_locally_relevant_dofs.split_by_block(dofs_per_block);

  this->pcout << "Number of degrees of freedom: " << this->dof_handler.n_dofs()
              << " (" << Patterns::Tools::to_string(dofs_per_block) << ")"
              << std::endl;

  this->constraints.clear();
  this->constraints.reinit(non_blocked_locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(this->dof_handler, this->constraints);



  for (const auto &id : this->dirichlet_ids)
    VectorTools::interpolate_boundary_values(*this->mapping,
                                             this->dof_handler,
                                             id,
                                             /*this->dirichlet_boundary_condition*/BoundaryValues<dim>(),
                                             this->constraints,
											 this->fe->component_mask(velocity));

  this->constraints.close();


  TrilinosWrappers::BlockSparsityPattern dsp(locally_owned_dofs,
                                             locally_owned_dofs,
                                             locally_relevant_dofs,
                                             this->mpi_communicator);

  DoFTools::make_sparsity_pattern(this->dof_handler,
                                  dsp,
                                  this->constraints,
                                  false,
                                  Utilities::MPI::this_mpi_process(
                                    this->mpi_communicator));

  // SparsityTools::distribute_sparsity_pattern(dsp,
  //                                            locally_owned_dofs,
  //                                            mpi_communicator,
  //                                            locally_relevant_dofs);
  dsp.compress();

  system_block_matrix.reinit(dsp);
  preconditioner_matrix.reinit(dsp);

  block_solution.reinit(locally_owned_dofs, this->mpi_communicator);
  system_block_rhs.reinit(locally_owned_dofs, this->mpi_communicator);

  locally_relevant_block_solution.reinit(locally_owned_dofs,
                                         locally_relevant_dofs,
                                         this->mpi_communicator);

  this->error_per_cell.reinit(this->triangulation.n_active_cells());

  // Now call anything that may be needed
  this->setup_system_call_back();
}



template <int dim>
void
BaseBlockProblem<dim>::assemble_system_one_cell(
  const typename DoFHandler<dim>::active_cell_iterator &,
  ScratchData &,
  CopyData &)
{
  Assert(false, ExcPureFunctionCalled());
}



template <int dim>
void
BaseBlockProblem<dim>::copy_one_cell(const CopyData &copy)
{
  this->constraints.distribute_local_to_global(copy.matrices[0],
                                               copy.vectors[0],
                                               copy.local_dof_indices[0],
                                               system_block_matrix,
                                               system_block_rhs);

  this->constraints.distribute_local_to_global(copy.matrices[1],
                                                 copy.local_dof_indices[0],
                                                 preconditioner_matrix);

}



template <int dim>
void
BaseBlockProblem<dim>::assemble_system()
{
  TimerOutput::Scope timer_section(this->timer, "assemble_system");
  QGauss<dim>        quadrature_formula(this->fe->degree + 1);
  QGauss<dim - 1>    face_quadrature_formula(this->fe->degree + 1);

  ScratchData scratch(*this->mapping,
                      *this->fe,
                      quadrature_formula,
                      update_values | update_gradients |
                        update_quadrature_points | update_JxW_values,
                      face_quadrature_formula,
                      update_values | update_quadrature_points |
                        update_JxW_values);

  CopyData copy(this->fe->n_dofs_per_cell());

  auto worker = [&](const auto &cell, auto &scratch, auto &copy) {
    assemble_system_one_cell(cell, scratch, copy);
  };

  auto copier = [&](const auto &copy) { copy_one_cell(copy); };

  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  WorkStream::run(CellFilter(IteratorFilters::LocallyOwnedCell(),
                             this->dof_handler.begin_active()),
                  CellFilter(IteratorFilters::LocallyOwnedCell(),
                             this->dof_handler.end()),
                  worker,
                  copier,
                  scratch,
                  copy);


  //Linear operators
//  auto A  = linear_operator< TrilinosWrappers::MPI::Vector >( stokes_matrix.block(0,0) );
//  auto Bt = linear_operator< TrilinosWrappers::MPI::Vector >( stokes_matrix.block(0,1) );
//  auto B     = linear_operator< TrilinosWrappers::MPI::Vector >( stokes_matrix.block(1,0) );
//  auto ZeroP = linear_operator< TrilinosWrappers::MPI::Vector >( stokes_matrix.block(1,1) );

//  const auto &A = system_block_matrix.block(0, 0);
//  const auto &B = system_block_matrix.block(1, 0);
//
//  const auto op_A = linear_operator(A);
//
//  ReductionControl reduction_control_A(2000, 1.0e-5, 1.0e-10);
//  SolverCG<Vector<double>>    solver_A(reduction_control_A);
//
//  const auto op_A_inv = inverse_operator(op_A, solver_A,PreconditionIdentity());
//
//  const auto op_B = linear_operator(B);
//  const auto op_Bt= transpose_operator(op_B);
//
//  const auto op_S = op_Bt*op_A_inv*op_B;

//  ReductionControl reduction_control_S(2000, 1.0e-5, 1.0e-10);
//  SolverCG<Vector<double>>    solver_S(reduction_control_S);
//  const auto preconditioner_S =
//    inverse_operator(op_sS, solver_aS,PreconditionIdentity());

  system_block_matrix.compress(VectorOperation::add);
  preconditioner_matrix.compress(VectorOperation::add);
  system_block_rhs.compress(VectorOperation::add);
}



template <int dim>
void
BaseBlockProblem<dim>::solve()
{
  TimerOutput::Scope timer_section(this->timer, "solve");
  // SolverCG<LA::MPI::BlockVector> solver(solver_control);
  // LA::MPI::PreconditionAMG  amg;
  // amg.initialize(system_matrix);
  // solver.solve(system_matrix, solution, system_rhs, amg);
  // constraints.distribute(solution);
  // locally_relevant_solution = solution;
}



template class BaseBlockProblem<1>;
template class BaseBlockProblem<2>;
template class BaseBlockProblem<3>;
