/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

#include <omp.h>
#include <iostream>

using idx_t = faiss::idx_t;

// Method to print the reconstructed vectors
void print_reconstructed_vectors(float* recons, idx_t ni, int d, idx_t i0 = 0) {
    // Loop through each vector
    for (idx_t i = 0; i < ni; ++i) {
        std::cout << "Vector " << i + i0 << ": "; // Print the original index
        for (int j = 0; j < d; ++j) {
            std::cout << recons[i * d + j] << " "; // Print each element of the vector
        }
        std::cout << "\n"; // Newline after printing each vector
    }
}


int main() {
    
    // CHEF NOTE: omp_set_num_threads() works here but not from python custom swig
    //      - both on release and debug
    omp_set_num_threads(4);
    std::cout << "set threads done\n" << std::endl; 

    int d = 128;      // dimension
    int nb = 10000; // database size
    int nq = 1000000;  // nb of queries

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    int nlist = 2;
    int k = 4;

    faiss::IndexFlatL2 quantizer(d); // the other index
    faiss::IndexIVFFlat index(&quantizer, d, nlist);
    assert(!index.is_trained);
    index.train(nb, xb);
    assert(index.is_trained);
    faiss::ArrayInvertedLists* before_add_invlists = dynamic_cast<faiss::ArrayInvertedLists*>(index.invlists);
    std::cout << "index.invlists.nlist: " << index.invlists->nlist << '\n';
    index.add(nb, xb);

    { // test crack
        
        // Assuming index.invlists is a pointer to an InvertedLists
        faiss::InvertedLists* invlists = index.invlists;  // Example assignment

        // Use typeid to check the type of the object
        if (typeid(*invlists) == typeid(faiss::ArrayInvertedLists)) {
            std::cout << "The object is of type ArrayInvertedLists." << std::endl;
        } else if (typeid(*invlists) == typeid(faiss::InvertedLists)) {
            std::cout << "The object is of type InvertedLists." << std::endl;
        } else {
            std::cout << "Unknown type!" << std::endl;
        }
        
         // Use dynamic_cast to check if it's an instance of ArrayInvertedLists
        faiss::ArrayInvertedLists* arrayInvertedLists = dynamic_cast<faiss::ArrayInvertedLists*>(invlists);

        // if (arrayInvertedLists) {
        //     // If the cast is successful, we can call add_empty_list()
        //     std::cout << "Adding empty list to ArrayInvertedLists." << std::endl;
        //     arrayInvertedLists->add_empty_list();  // Call the method only if it's ArrayInvertedLists
        // } else {
        //     std::cout << "The object is not of type ArrayInvertedLists." << std::endl;
        // }
   
        // Print the value of nlist before and after adding an empty list
        std::cout << "index.nlist: " << index.nlist << '\n';
        std::cout << "index.invlists.nlist: " << index.invlists->nlist << '\n';
        // index.invlists->add_empty_list();  // Add an empty list
        arrayInvertedLists->add_empty_list(); // need to dynamic cast because add_empty_list() only implemented for ArrayInvertedLists
        // CHEF TODO: you need ot handle this case. Should .add_empty_list() be part of the index?
        // -- probably I need to create new type of dynamic index to support this stuff, for now hack it away
        index.nlist++;
        std::cout << "index.nlist: " << index.nlist << '\n';
        std::cout << "index.invlists.nlist: " << index.invlists->nlist << '\n';

        // Print the size of each list in ids
        std::cout << "Size of each list in ids: \n";
        for (size_t i = 0; i < index.nlist; ++i) {
            // std::cout << "List " << i << " size: " << index.invlists->ids[i].size() << '\n';
            std::cout << "List " << i << '\n';
            std::cout << "-- index.invlists->list_size(list_no) (official): " << index.invlists->list_size(i) << '\n';
        }

    }

    /*
    Even though .add_empty_list() works as a method on ArrayInvertedLists. 
    
    You still need to handle (not done yet):
        - index.nlist ( doesn't know an invlist was added )
        - index.quantizer ( doesn't have centroid for new invlist )
            - if you don't change quantizer, there can never be assignments to the new invlist

    Need to test:
        - if swig automatically added the add_empty_list() method to python so I can continue with prototype

    Tested (Success):
        - adding elements to this new invlists // SUCCESS <<
            - works with add_core() [add preassigned]
            - if you don't change quantizer, it won't work if you don't have preassigned
                - won't know what to compare vectors against to put them on this new invlist

    more TODO:
        - move test from here to a test file, to have as reference
        - 
    */

    { // test adding elements to this new invlist
    // SUCCESS <<
    // Example setup (assuming d = 5 and ln = 2)
        int ln = 2;
        int d = 5;  // Dimension of vectors
        float* data_to_add = new float[ln * d]; // Allocate space for 2 vectors of dimension 5
        idx_t* assignments = new idx_t[ln];     // Allocate space for vector IDs

        for (int i = 0; i < ln; i++) {
            for (int j = 0; j < d; j++) {
                data_to_add[d * i + j] = distrib(rng);  // Fill with random values
            }
            data_to_add[d * i] += i / 1000.;  // Modify first element of each vector
        }

        // Assign them to list 2
        for (int i = 0; i < ln; i++) {
            assignments[i] = 2;  // Assign vectors to the 2nd list
        }

        // Assuming you want to leave precomputed_idx as nullptr
        idx_t* precomputed_idx = nullptr;

        // Call add_core to add the vectors to the index
        index.add_core(ln, data_to_add,  precomputed_idx, assignments, nullptr);

        // Clean up allocated memory
        delete[] data_to_add;
        delete[] assignments;

        // Print the updated size of each list in the inverted lists
        std::cout << "[NEW] Size of each list in ids: \n";
        for (size_t i = 0; i < index.nlist; ++i) {
            std::cout << "List " << i << '\n';
            std::cout << "-- index.invlists->list_size(list_no) (official): " 
                    << index.invlists->list_size(i) << '\n';
        }

    }

    // { // test adding new quantizer
                
    //     // Create an IndexIVF object (assumed to have been initialized)
    //     faiss::IndexFlatL2 quantizer(d); // Example quantizer
    //     faiss::IndexIVFFlat index(&quantizer, d, 100); // nlist = 100 (number of inverted lists)

    //     // Train and add vectors to the index (for this example, we assume the index is trained)
    //     index.train(n, X);
    //     index.add(n, X);
    // }


    // { // test if quantizer works
    //     idx_t i0 = 0;  // Starting index for reconstruction
    //     idx_t ni = index.nlist - 1; // Number of vectors to reconstruct (initial value)
    //     float* recons = new float[ni * d]; // Allocate memory for the first reconstruction

    //     // Perform the reconstruction with the first value of ni
    //     std::cout << "index.quantizer->ntotal " << index.quantizer->ntotal << '\n';

    //     std::cout << "Reconstructing " << ni << " vectors from index " << i0 << '\n';
    //     index.quantizer->reconstruct_n(i0, ni, recons);  // Call without trying to print a return value
    //     print_reconstructed_vectors(recons, ni, d, i0);

    //     // Now, change ni and reuse the same recons pointer
    //     ni = index.nlist; // Change ni to the desired number of vectors to reconstruct

    //     // Optionally, if you want to overwrite the recons buffer with the new reconstruction, you can:
    //     recons = new float[ni * d]; // Reallocate memory if needed

    //     std::cout << "Reconstructing " << ni << " vectors from index " << i0 << '\n';
    //     index.quantizer->reconstruct_n(i0, ni, recons);  // Call without trying to print a return value
    //     print_reconstructed_vectors(recons, ni, d, i0);

    //     // Don't forget to free the allocated memory after use
    //     delete[] recons;
    // }

    { // search xq
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];

        index.search(nq, xq, k, D, I);

        printf("I=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5f ", D[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    delete[] xb;
    delete[] xq;

    return 0;
}
