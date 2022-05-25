#define PY SSIZE T CLEAN
#include <Python.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

/* INTERNAL START */
#define EPSILON 0.001
int K; /* num of centroids */
int D; /* num of coordinates */
int N; /* num of points */
int ITER_MAX;

typedef struct Point{
    float *coordinates;
    int centroid_id;
} Point;

typedef struct Centroid{
    float *coordinates;
    float *prev_coords;
    int centroid_id;
    float *sum;
    float cnt;
} Centroid;


void sum_arrays(float p1[], float p2[]){
    int i;
    for (i=0 ; i<D ; ++i){
        p1[i] += p2[i];
    }
}

Centroid* init_centroids(Point points[]){
    int i;
    Centroid centroid;
    Centroid* centroids = (Centroid *)calloc(K, sizeof(Centroid));
    if (centroids==NULL){
        printf("An Error Has Occurred\n");
        exit(1);
    }
    for (i = 0; i < K; ++i) {
        float* c_coords =  (float*) calloc(D, sizeof(float));
        if (c_coords==NULL){
            printf("An Error Has Occurred\n");
            exit(1);
        }
        sum_arrays(c_coords, points[i].coordinates);
        centroid.coordinates = c_coords;
        centroid.sum = (float*) calloc(D, sizeof(float ));
        if (centroid.sum==NULL){
            printf("An Error Has Occurred\n");
            exit(1);
        }
        centroid.prev_coords = (float*) calloc(D, sizeof(float ));
        if (centroid.prev_coords==NULL){
            printf("An Error Has Occurred\n");
            exit(1);
        }
        centroid.centroid_id = i;
        centroid.cnt = 0;
        centroids[i] = centroid;
    }
    return centroids;
}

float euclidean_norm(float p1[], float p2[]){
    float sum = 0;
    int i;
    for (i = 0; i < D; ++i) {
        sum += (float)pow((p1[i]-p2[i]),2);
    }
    return (float)sqrt((double)sum);
}


void assign_points_to_cluster(Point points[], Centroid centroids[]){
    int i, j;
    Point* p;
    Centroid* min_c;
    Centroid* c;
    float tmp_dist;
    float min_dist;
    for (i = 0; i < N ; ++i) {
        p = &points[i];
        min_dist = FLT_MAX;
        min_c = centroids;
        for (j = 0; j < K; ++j) {
            c = &centroids[j];
            tmp_dist = euclidean_norm(c->coordinates, p->coordinates);
            min_c = (tmp_dist < min_dist) ? c : min_c;
            min_dist = (tmp_dist<min_dist) ? tmp_dist : min_dist;
        }
        p->centroid_id = min_c->centroid_id;
        min_c->cnt += 1;
        sum_arrays(min_c->sum, p->coordinates);
    }
}


int update_centroids(Centroid centroids[]){
    int i, j;
    int epsilon_check = 1;
    for (i = 0; i < K; ++i) {
        Centroid * c = centroids + i;
        free(c->prev_coords);
        c->prev_coords = c->coordinates;
        c->coordinates = (float*) calloc(D, sizeof(float ));
        if (c->coordinates==NULL){
            printf("An Error Has Occurred\n");
            exit(1);
        }
        for (j = 0; j < D; ++j) {
            *(c->coordinates+j) = (c->sum[j] / c->cnt);
        }
        c->cnt = 0;
        free(c->sum);
        c->sum = (float*) calloc(D, sizeof(float ));
        if (c->sum==NULL){
            printf("An Error Has Occurred\n");
            exit(1);
        }
        if (euclidean_norm(c->coordinates, c->prev_coords)>= EPSILON){
            epsilon_check = 0;
        }
    }
    return epsilon_check;
}
void free_points(Point points[]){
    int i;
    for (i = 0; i < N; ++i) {
        free(points[i].coordinates);
    }
    free(points);
}


void free_centroids(Centroid centroids[]){
    int i;
    for (i = 0; i < K; ++i) {
        free(centroids[i].coordinates);
        free(centroids[i].sum);
        free(centroids[i].prev_coords);
    }
    free(centroids);
}

Centroid* kmeans(Point points[], Centroid centroids[]){
    int epsilon_check = 0;
    int i =0;
    while (!epsilon_check && i<ITER_MAX){
        assign_points_to_cluster(points, centroids);
        epsilon_check = update_centroids(centroids);
        i++;
    }
    return centroids;
}

/* INTERNAL END */

/* Expected params:
 *  1. Num of coordinates
 *  2. Num of centroids
 *  3. Num of points
 *  4. Max iteration
 *  5. All the coordinates one by one, centroids first, then points. Corresponding to the values above.
 * */
static PyObject* fit(PyObject *self, PyObject *args) {
    PyObject *obj;
    int i, j;
    Centroid* centroids;
    float* c_coords;
    PyObject *next;
    double curr;
    Point* points;
    Centroid* res_centroids;
    float* res;
    PyObject* py_res;
    PyObject* python_float;

    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }
    PyObject *iter = PyObject_GetIter(obj);
    if (!iter) {
        return NULL;
    }
    for (i = 0; i < 4; i++) { /* Fetching the parameters: num of centroids and num of coordinates */
        next = PyIter_Next(iter);
        if (!next) {  /* nothing left in the iterator, we expect at least the parameters */
            return NULL;
        }
        if (!PyFloat_Check(next)) { /* verifying the value of the current item is type float */
            return NULL;
        }
        curr = PyFloat_AsDouble(next);
        switch (i) {
            case 0:
                D = (int)curr;
                break;
            case 1:
                K = (int)curr;
                break;
            case 2:
                N = (int)curr;
                break;
            case 3:
                ITER_MAX = (int)curr;
                break;
        }
    }

    /* Creating centroids out of the given coordinates */
    centroids = (Centroid*)calloc(K, sizeof(Centroid));
    for (i = 0; i < K; i++){ /* fetch the centroids */
        c_coords =  (float*) calloc(D, sizeof(float));
        for(j = 0; j < D; j++){
            next = PyIter_Next(iter);
            if (!next) {  /* nothing left in the iterator, we expect at least the parameters */
                return NULL;
            }
            if (!PyFloat_Check(next)) { /* verifying the value of the current item is type float */
                return NULL;
            }
            curr = PyFloat_AsDouble(next);
            c_coords[j] = (float)curr;
        }
        centroids[i].coordinates = c_coords;
        centroids[i].sum = (float*) calloc(D, sizeof(float ));
        if (centroids[i].sum==NULL){
            exit(1);
        }
        centroids[i].prev_coords = (float*) calloc(D, sizeof(float ));
        if (centroids[i].prev_coords==NULL){
            exit(1);
        }
        centroids[i].centroid_id = i;
        centroids[i].cnt = 0;
    }
    /* Creating points out of the given coordinates */
    points = (Point*)calloc(N, sizeof(Point));
    for(i = 0; i < N; i++) { /* fetch points */
        c_coords =  (float*) calloc(D, sizeof(float));
        for(j = 0; j < D; j++){
            next = PyIter_Next(iter);
            if (!next) {  /* we expect to have all coordinates for all points */
                return NULL;
            }
            if (!PyFloat_Check(next)) { /* verifying the value of the current item is type float */
                return NULL;
            }
            curr = PyFloat_AsDouble(next);
            c_coords[j] = (float)curr;
        }
        points[i].coordinates = c_coords;
    }
    res_centroids = (Centroid *)calloc(K, sizeof(Centroid));
    res_centroids = kmeans(points, centroids);
    res = (float*) calloc(K * D, sizeof(float));
    for(i = 0; i < K; i++){
        for(j = 0; j < D; j++){
            res[i * D + j] = res_centroids[i].coordinates[j];
        }
    }
    py_res = PyList_New(D * K);
    for (i = 0; i < D * K; ++i)
    {
        python_float = Py_BuildValue("f", res[i]);
        PyList_SetItem(py_res, i, python_float);
    }
    free_points(points);
    free_centroids(centroids);
    free(res);
    //free_centroids(res_centroids);
    return py_res;
}


/*This array tells Python what methods this module has. We will use it in the next structure*/

static PyMethodDef capiMethods [] = {
        {"fit",
         (PyCFunction) fit,
         METH_VARARGS,
         PyDoc_STR("Kmeans function, implemented in C")},
         {NULL, NULL, 0, NULL}
};

 static struct PyModuleDef moduledef = {
         PyModuleDef_HEAD_INIT,
         "mykmeanssp",
         NULL,
         -1,
         capiMethods
 };

PyMODINIT_FUNC
PyInit_mykmeanssp (void){
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}
