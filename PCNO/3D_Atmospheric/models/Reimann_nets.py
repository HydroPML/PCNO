import torch
from itertools import combinations_with_replacement

def tensor_product(omega_i, omega_j):
    return torch.einsum('ab,cd->abcd', omega_i, omega_j)

def create_basis_2forms(n=4):
    basis_2forms = []
    for i, j in combinations_with_replacement(range(n), 2):
        if i < j:
            omega = torch.zeros((n, n), dtype=torch.float64)
            omega[i, j] = 1.0
            omega[j, i] = -1.0
            basis_2forms.append(omega)
    return basis_2forms

def build_basis(n=4):
    basis_2forms = create_basis_2forms(n)
    symmetric_products = []

    for i in range(len(basis_2forms)):
        for j in range(i, len(basis_2forms)):
            if i < j:
                sym_prod = 0.5 * (tensor_product(basis_2forms[i], basis_2forms[j]) +
                                  tensor_product(basis_2forms[j], basis_2forms[i]))
            else:
                sym_prod = tensor_product(basis_2forms[i], basis_2forms[j])
            symmetric_products.append(sym_prod)
    return symmetric_products