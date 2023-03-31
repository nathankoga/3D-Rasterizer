// https://www.youtube.com/shorts/QuVRm1S6Uyk
// Youtube link to video

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define NORMALS


double C441(double f)
{
    return ceil(f-0.00001);
}

double F441(double f)
{
    return floor(f+0.00001);
}

typedef struct 
{
    double lightDir[3]; // The direction of the light source
    double Ka;           // The coefficient for ambient lighting.
    double Kd;           // The coefficient for diffuse lighting.
    double Ks;           // The coefficient for specular lighting.
    double alpha;        // The exponent term for specular lighting.
} LightingParameters;

typedef struct
{
    double          A[4][4];     // A[i][j] means row i, column j
} Matrix;

typedef struct
{
    double          near, far;
    double          angle;
    double          position[3];
    double          focus[3];
    double          up[3];
} Camera;

typedef struct
{
   double         X[3];
   double         Y[3];
   double         Z[3];
   double         color[3][3]; // color[2][0] is for V2, red channel
   double         shading[3];
#ifdef NORMALS
   double         normals[3][3]; // normals[2][0] is for V2, x-component
#endif
} Triangle;


typedef struct
{
   int numTriangles;
   Triangle *triangles;
} TriangleList;


typedef struct {
    int width;
    int height;
    int totalPixels;  
    unsigned char* buff;
    double* zBuffer;
} Image;


void
PrintMatrix(Matrix m)
{
    for (int i = 0 ; i < 4 ; i++)
    {
        printf("(%.7f %.7f %.7f %.7f)\n", m.A[i][0], m.A[i][1], m.A[i][2], m.A[i][3]);
    }
}


Matrix
ComposeMatrices(Matrix M1, Matrix M2)
{
    Matrix m_out;
    for (int i = 0 ; i < 4 ; i++)
        for (int j = 0 ; j < 4 ; j++)
        {
            m_out.A[i][j] = 0;
            for (int k = 0 ; k < 4 ; k++)
                m_out.A[i][j] += M1.A[i][k]*M2.A[k][j];
        }
    return m_out;
}


double SineParameterize(int curFrame, int nFrames, int ramp)
{  
    int nNonRamp = nFrames-2*ramp;
    double height = 1./(nNonRamp + 4*ramp/M_PI);
    if (curFrame < ramp)
    {
        double factor = 2*height*ramp/M_PI;
        double eval = cos(M_PI/2*((double)curFrame)/ramp);
        return (1.-eval)*factor;
    }
    else if (curFrame > nFrames-ramp)
    {        
        int amount_left = nFrames-curFrame;
        double factor = 2*height*ramp/M_PI;
        double eval =cos(M_PI/2*((double)amount_left/ramp));
        return 1. - (1-eval)*factor;
    }        
    double amount_in_quad = ((double)curFrame-ramp);
    double quad_part = amount_in_quad*height;
    double curve_part = height*(2*ramp)/M_PI;
    return quad_part+curve_part;
} 

Camera       
GetCamera(int frame, int nframes)
{            
    double t = SineParameterize(frame, nframes, nframes/10);
    Camera c;
    c.near = 5;
    c.far = 200;
    c.angle = M_PI/6;
    c.position[0] = 40*sin(2*M_PI*t);
    c.position[1] = 40*cos(2*M_PI*t);
    c.position[2] = 40;
    c.focus[0] = 0; 
    c.focus[1] = 0; 
    c.focus[2] = 0;
    c.up[0] = 0;    
    c.up[1] = 1;    
    c.up[2] = 0;    
    return c;       
}


LightingParameters 
GetLighting(Camera c)
{
    LightingParameters lp;
    lp.Ka = 0.3;
    lp.Kd = 0.7;
    lp.Ks = 2.8;
    lp.alpha = 50.5;
    lp.lightDir[0] = c.position[0]-c.focus[0];
    lp.lightDir[1] = c.position[1]-c.focus[1];
    lp.lightDir[2] = c.position[2]-c.focus[2];
    double mag = sqrt(lp.lightDir[0]*lp.lightDir[0]
                    + lp.lightDir[1]*lp.lightDir[1]
                    + lp.lightDir[2]*lp.lightDir[2]);
    if (mag > 0)
    {
        lp.lightDir[0] /= mag;
        lp.lightDir[1] /= mag;
        lp.lightDir[2] /= mag;
    }

    return lp;
}


void 
TransformPoint(Matrix m, const double *ptIn, double *ptOut)
{  
    ptOut[0] = ptIn[0]*m.A[0][0]
             + ptIn[1]*m.A[1][0]
             + ptIn[2]*m.A[2][0]
             + ptIn[3]*m.A[3][0];
    ptOut[1] = ptIn[0]*m.A[0][1]
             + ptIn[1]*m.A[1][1]
             + ptIn[2]*m.A[2][1]
             + ptIn[3]*m.A[3][1];
    ptOut[2] = ptIn[0]*m.A[0][2]
             + ptIn[1]*m.A[1][2]
             + ptIn[2]*m.A[2][2]
             + ptIn[3]*m.A[3][2];
    ptOut[3] = ptIn[0]*m.A[0][3]
             + ptIn[1]*m.A[1][3]
             + ptIn[2]*m.A[2][3]
             + ptIn[3]*m.A[3][3];
}

void 
CrossProduct(double *A, double *B, double *vec){
   vec[0] = A[1]*B[2] - A[2]*B[1]; 
   vec[1] = B[0]*A[2] - A[0]*B[2]; 
   vec[2] = A[0]*B[1] - A[1]*B[0]; 
}

double 
DotProduct(double *A, double *B){
   double product = A[0]*B[0] +
                    B[1]*A[1] +
                    A[2]*B[2]; 
   return product;
}

void
Normalize(double *vec){
   double norm = sqrt(vec[0]*vec[0] +
                  vec[1]*vec[1] +
                  vec[2]*vec[2]);
   vec[0] = vec[0] / norm;
   vec[1] = vec[1] / norm;
   vec[2] = vec[2] / norm;
}


Matrix
GetCameraTransform(Camera c)
{   
   Matrix rv;

   double temp[3];
   double u[3]; double v[3];  // initialize, set with CrossProduct later
   double O[3] = {c.position[0], c.position[1], c.position[2]};  // O == camera position
   double w[3] = {(O[0]-c.focus[0]), (O[1] - c.focus[1]), (O[2] - c.focus[2])};  // w = O - focus
   double t[3] = {0-O[0], 0 - O[1], 0 - O[2]};
   
   /*u = */ CrossProduct(c.up, w, u);  // u = up x w  (w == O-focus)
   /*v = */ CrossProduct(w, u, v);   // v = w x u
   Normalize(u); Normalize(w); Normalize(v);
   // Matrix cameraFrame;

   for (int row =0; row <3; row++){
      rv.A[row][0] = u[row];
      rv.A[row][1] = v[row];
      rv.A[row][2] = w[row];
      rv.A[row][3] = 0;

   }
  
   rv.A[3][0] = DotProduct(u, t);
   rv.A[3][1] = DotProduct(v, t);
   rv.A[3][2] = DotProduct(w, t);
   rv.A[3][3] = 1;

   return rv;
}


Matrix GetViewTransform(Camera c)  // cartesian to camera
{
   Matrix m;
   double alpha = 1 / tan(c.angle / 2);
   
   for (int row = 0; row < 4; row++){
      for (int col = 0; col <4; col++){
         // check for the specific cases, else, initialize matrix
         if (row == 0 && col == 0)
            m.A[0][0] = alpha;
         else if (row == 1 && col == 1)
            m.A[1][1] = alpha;
         else if (row == 2 && col == 2)
            m.A[2][2] = (c.far+c.near)/(c.far-c.near);
         else if (row == 2 && col == 3)
            m.A[2][3] = -1;
         else if (row == 3 && col == 2)
            m.A[3][2] = 2*(c.far*c.near)/(c.far-c.near);
         else
            m.A[row][col] = 0;
      }
   }
   // PrintMatrix(m);
   // printf("\n");
   return m;   
}


Matrix
GetDeviceTransform(Camera c, Image *img)
{   
   Matrix rv;
   double m = img->height;
   double n = img->width;
   for (int row = 0; row < 4; row++){
      for (int col = 0; col <4; col++){
         rv.A[row][col] = 0;
      }
   }
   rv.A[0][0] = n/2;
   rv.A[1][1] = m/2;
   rv.A[2][2] = 1;
   rv.A[3][0] = n/2;
   rv.A[3][1] = m/2;
   rv.A[3][3] = 1;
   // PrintMatrix(rv);
   // printf("\n");
   
   return rv;
}


Image *makeBaseImage(int width, int height){ 
   Image *image = (Image *) malloc(sizeof(Image)); 
   image->width = width;
   image->height = height;
   image->totalPixels = width*height;
   image->buff = (unsigned char*) malloc(image->totalPixels*3 * sizeof(unsigned char)) ;
   image->zBuffer = (double *) malloc(image->totalPixels*sizeof(double));
   for (int i = 0; i<image->totalPixels; i++){
      image->zBuffer[i] = -1;  // set z for every pixel to -1
   }

   return image;
}

void InitializeScreen(Image *img){
   // reset each screen to black, and reset zBuffer to -1
   unsigned char black = 0;
   for (int pixel = 0; pixel < img->totalPixels; pixel++){
      img->buff[pixel*3] = black;
      img->buff[pixel*3+1] = black;
      img->buff[pixel*3+2] = black;
   }
   
   for (int i = 0; i<img->totalPixels; i++){
      img->zBuffer[i] = -1;  
   }
}


FILE *createPNM (Image *img, char fileName[]){
    FILE* imgFile;
    imgFile = fopen(fileName, "w");     
    int width = img->width;
    int height = img->height;
    fprintf(imgFile, "%s", "P6\n");
    fprintf(imgFile, "%d%s%d%s", width, " ", height, "\n");
    fprintf(imgFile, "%d%s", 255, "\n");
    for (int pixel = 0; pixel < img->totalPixels; pixel++){
	    fprintf(imgFile, "%c", img->buff[pixel*3]);
        fprintf(imgFile, "%c", img->buff[pixel*3+1]);
        fprintf(imgFile, "%c", img->buff[pixel*3+2]);
    } 
    return imgFile;
}


double LERP(double t, double fA, double fB){
   double fX = fA + t*(fB-fA);
   return fX;
}


double proportion(double A, double X, double B){
   double t = B == A ? 1 : (X-A) / (B-A);
   return t;
}


void assignPixels(Image *img, Triangle *triangle, double colMin, double colMax, int rowMin, int rowMax, double tLeft, double tRight, int top, int left, int right, int swapped){
    // rounded and unrounded boundaries both necessary for interpolation
   float roundedMin = C441(colMin);
   float roundedMax = F441(colMax);

   // colors of left and right
   if (swapped){  // tag for whether or not the left and right values were swapped
      int temp = left;
      left = right;
      right = temp;
   }
   double shadeLeft = LERP(tLeft, triangle->shading[top], triangle->shading[left]); double shadeRight = LERP(tRight, triangle->shading[top], triangle->shading[right]); 
   double rLeft = LERP(tLeft, triangle->color[top][0], triangle->color[left][0]); double rRight= LERP(tRight, triangle->color[top][0], triangle->color[right][0]);
   double gLeft = LERP(tLeft, triangle->color[top][1], triangle->color[left][1]); double gRight = LERP(tRight, triangle->color[top][1], triangle->color[right][1]);
   double bLeft = LERP(tLeft, triangle->color[top][2], triangle->color[left][2]); double bRight = LERP(tRight, triangle->color[top][2], triangle->color[right][2]);
   
   // z of left and right
   double zLeft = LERP(tLeft, triangle->Z[top], triangle->Z[left]);
   double zRight= LERP(tRight, triangle->Z[top], triangle->Z[right]);      

    for (int row= rowMin; row < rowMax && row < img->height; row++){  
        for (int col= roundedMin; col <= roundedMax && col < img->width; col++){  // adjusted to <= as we include the ceiling
            
            double t = proportion(colMin, col, colMax);  // find the proportion of current col vs left and right
            double z = LERP(t, zLeft, zRight);  // use t to find z, r, g, and b
            double red = LERP(t, rLeft, rRight);
            double green= LERP(t, gLeft, gRight);
            double blue = LERP(t, bLeft, bRight);
            double shade = LERP(t, shadeLeft, shadeRight);

            int newRow = img->height-row-1;
            if (col < 0) col = 0; 
            if (col > img->width) col = img->width-1;
            
            if ((0 <= col) && (col <img->width) && (0 <= newRow) && (newRow < img->height)){
               if (z > img->zBuffer[img->width*newRow + col]){  // z-buffer algorithm: if z is closer, change color
                  img->buff[img->width*newRow*3 + col*3] = (unsigned char)C441(fmin(red*shade, 1)*255);
                  img->buff[img->width*newRow*3 + col*3+1] = (unsigned char)C441(fmin(green*shade, 1)*255);
                  img->buff[img->width*newRow*3 + col*3+2] = (unsigned char)C441(fmin(blue*shade, 1)*255);
                  img->zBuffer[img->width*newRow + col] = z;
               } 
            }
        }
    }
}


void RasterizeTriangle(Triangle *triangle, Image *img){
   int topVertex, botVertex; // to store index numbers
    
   double top =  triangle->Y[0];  // arbitrarily set top and bot values to compare
   double bot =  triangle->Y[0];  
   int idx1 = 0;  // initialize comparison variables to track indices
   int idx2 = 0; 


   for (int i = 0; i < 3; i++){  // determine the points   
      if (i == 0){
         idx1 = 1;  // randomly set the other comparisons
         idx2 = 2;
      }
      if (i == 1){
         idx1 = 0;
         idx2 = 2;
      }
      if (i == 2){
         idx1 = 0;
         idx2 = 1;  
      }

      // find top vertex: in a tie, it will be farthest left
      if (triangle->Y[i] >= top) {  
         if (triangle->Y[i] == triangle->Y[idx1]){  // flat top
            if (triangle->X[i] < triangle->X[idx1]){
               topVertex = i;
            }
            else {
               topVertex = idx1;
            }
         }
         else if (triangle->Y[i] == triangle->Y[idx2]) {  // flat top
            if (triangle->X[i] < triangle->X[idx2]){
               topVertex = i;
            }
            else {
               topVertex = idx2;
            }

         }
         else {  // i has highest y value, with no contesting vertices
            topVertex = i;
         }
         top = triangle->Y[topVertex];
      } 

      // find bot vertex: in a tie, will be farthest left
      if (triangle->Y[i] <= bot){
         if (triangle->Y[i] == triangle->Y[idx1]){  // flat bot
            if (triangle->X[i] < triangle->X[idx1]){
               botVertex = i;
            }
            else {
               botVertex = idx1;
            }
         }
         else if (triangle->Y[i] == triangle->Y[idx2]) {  // flat bot
            if (triangle->X[i] < triangle->X[idx2]){
               botVertex = i;
            }
            else {
               botVertex = idx2;
            }

         }
         else {  // i has highest y value, with no contesting vertices
            botVertex = i;
         }
         bot = triangle->Y[botVertex];

      }

   }
   // top & bot vertices identified
   
   if (((topVertex == 0) || (botVertex == 0)) && ((topVertex == 1) || (botVertex == 1))){ idx2 = 2;}
   else if (((topVertex == 0) || (botVertex == 0)) && ((topVertex == 2) || (botVertex == 2))){ idx2 = 1;}
   else {idx2 = 0;}
   
   // initialize specific x, y and z values for cleaner code
   double xBot = triangle->X[botVertex]; double yBot = triangle->Y[botVertex]; double zBot = triangle->Z[botVertex];
   double xTop= triangle->X[topVertex]; double yTop = triangle->Y[topVertex]; double zTop = triangle->Z[topVertex];
   double x2 = triangle->X[idx2]; double y2 = triangle->Y[idx2]; double z2 = triangle->Z[idx2];
   // initialize variables
   double mBotSlope, mTopSlope, mTopBot;
   double bBotSlope, bTopSlope, bTopBot;


   // create all slopes 
   if (xBot != x2){  // bot to idx2
      mBotSlope = (y2 -yBot)/ (x2 - xBot);  // m = (y2-y1) / (x2-x1) 
      bBotSlope = yBot - mBotSlope * xBot;  // b = y - mx
   }
   
   if (xTop != x2){  // top to idx2
      mTopSlope = (y2 -yTop)/ (x2 - xTop);
      bTopSlope = yTop - mTopSlope * xTop;
   }

   if (xTop != xBot){  // top to bot
      mTopBot= (yTop-yBot)/ (xTop- xBot);
      bTopBot= yTop - mTopBot * xTop;
   }

   int checker = (idx2 == botVertex) || (idx2  == topVertex) || (botVertex == topVertex);
   
   if (checker == 1){  // Seemingly "ERRORS" when a triangle is actually a line (two points are identical)
      printf("ERROR, NOT CORRECT: bot:(%d), top: (%d), idx2: (%d) \n", botVertex, topVertex, idx2);  
   }

   else {  // i.e. there are no lines (triangles with only two unique points)
      double leftCol, rightCol, temp, tempZ, tempT, tempV, tempY;
      double tLeft, tRight;  // we want to pass tLeft and tRight so that we can use those ratios to calculate colors and z values within assignment
      double zLeft, zRight;
      int swapped = 0;  // logic necessary for bottom half of an arbitrary triangle (if top half had to swap vertices)
         
         // Set up scanlines, based on specific criteria

      if ((yBot != y2) && (yTop != y2)){  // if arbitrary, meaning we have to switch in the middle
        
         // rasterize top half, using slope from top to bot & top to idx2
         for (int i = C441(y2); i<= F441(yTop); i++){
            swapped = 0;
            leftCol = xTop == xBot ? xBot : (i - bTopBot) / mTopBot;
            rightCol = xTop == x2 ? x2 : (i - bTopSlope) / mTopSlope;
            
            tLeft = proportion(yTop, i, yBot);
            tRight = proportion(yTop, i, y2);
            
            if (leftCol > rightCol){  // switch values
               temp = leftCol;
               leftCol = rightCol;
               rightCol = temp;
               
               tempT = tLeft;
               tLeft = tRight;
               tRight = tempT;
               swapped = 1; 
               
            }
            assignPixels(img, triangle, leftCol, rightCol, i, i+1, tLeft, tRight, topVertex, botVertex, idx2, swapped);
         } 
         // now rasterize bot half, using slope from top to bot & idx2 to bot
         for (int i = C441(yBot); i<= F441(y2); i++){
            swapped = 0;
            leftCol = xTop == xBot ? xBot : (i - bTopBot) / mTopBot;
            rightCol = xBot == x2 ? x2 : (i - bBotSlope) / mBotSlope;
            
            tLeft = proportion(yBot, i, yTop);
            tRight = proportion(yBot, i, y2);
           
            if (leftCol > rightCol){  
               temp = leftCol;
               leftCol = rightCol;
               rightCol = temp;
               
               tempT = tLeft;
               tLeft = tRight;
               tRight = tempT;
               swapped = 1;
            }
           
            assignPixels(img, triangle, leftCol, rightCol, i, i+1,tLeft, tRight, botVertex, topVertex, idx2, swapped);
         } // rasterize on bot completed
      }
      
      
      else if (y2 == yTop){  // if y2 == yTop, we have a strictly downwards pointing triangle
         
         for (int i = C441(yBot); i<= F441(yTop); i++){
            swapped = 0;
            leftCol = xTop == xBot ? xBot : (i - bTopBot) / mTopBot;
            rightCol = x2 == xBot ? xBot : (i - bBotSlope) / mBotSlope;
            
            tLeft = proportion(yBot, i, yTop);
            tRight = proportion(yBot, i, y2);

            if (leftCol > rightCol){  // switch values
               temp = leftCol;
               leftCol = rightCol;
               rightCol = temp;
               
               tempT = tLeft;
               tLeft = tRight;
               tRight = tempT;
               
               swapped = 1;
              
            }
            
            assignPixels(img, triangle, leftCol, rightCol, i, i+1, tLeft, tRight, botVertex, topVertex, idx2, swapped);
         } 
         
      }

      else if(y2 == yBot){// if y2 == yBot, we have a strictly upwards pointing triangle
         
         for (int i = C441(yBot); i<= F441(yTop); i++){
            swapped = 0;
            leftCol = xTop == xBot ? xBot : (i - bTopBot) / mTopBot;
            rightCol = xTop == x2 ? x2 : (i - bTopSlope) / mTopSlope;
            
            tLeft = proportion(yTop, i, yBot);
            tRight = proportion(yTop, i, y2);
            if (leftCol > rightCol){  // switch values
               temp = leftCol;
               leftCol = rightCol;
               rightCol = temp;
               
               tempT = tLeft;
               tLeft = tRight;
               tRight = tempT;
               
               swapped = 1;
              
            }

            assignPixels(img, triangle, leftCol, rightCol, i, i+1, tLeft, tRight, topVertex, botVertex, idx2, swapped);
         } 
      }
   }

}

double CalculatePhongShading(LightingParameters lp, double *viewDirection, double *normal){
   
   // Calculate Phong Shading for each vertex
   double LdotN = DotProduct(lp.lightDir, normal);
   double diffuse = lp.Kd*fmax(LdotN, 0);  // IS THIS CORRECT DIFFUSE
   
   double reflection[3];
   for (int i = 0; i<3; i++){
      reflection[i] = 2*LdotN*normal[i] - lp.lightDir[i];
   }
   
   double RdotV = DotProduct(reflection, viewDirection);
   double specular = lp.Ks * fabs(pow(fmax(0, RdotV), lp.alpha));
   double finalShading = specular + diffuse + lp.Ka;

   // printf("        View dir for pt is (%f, %f, %f)\n", viewDirection[0], viewDirection[1], viewDirection[2]);
   // printf("        Normal is (%f, %f, %f)\n", normal[0], normal[1], normal[2]);
   // printf("        LdotN is (%f)\n",LdotN); 
   // printf("    Diffuse is (%f)\n", diffuse);
   // printf("        Reflection vector R is (%f, %f %f)\n",reflection[0], reflection[1], reflection[2]); 
   // printf("        RdotV is (%f)\n", RdotV); 
   // printf("    Specular component is (%f)\n", specular);
   // printf("    Total value for vertex is (%f)\n", finalShading);
   return finalShading;
}


void TransformAndRenderTriangles(Camera c, TriangleList *tl, Image * img, LightingParameters lp){
   
    // get all transformations
    Matrix camTransform = GetCameraTransform(c);
    Matrix viewTransform = GetViewTransform(c);
    Matrix deviceTransform = GetDeviceTransform(c, img);
    Matrix m = ComposeMatrices(camTransform, viewTransform);
    m = ComposeMatrices(m, deviceTransform);

    // for (int i = 0; i < 10; i++){
    for (int i = 0; i < tl->numTriangles; i++){

        Triangle *newTriangle = malloc (sizeof(Triangle));
        // printf("Working on triangle %d\n", i);

        /*
        for (int v=0; v<3; v++){
            printf("      (%f, %f, %f), color = (%f, %f, %f)\n", tl->triangles[i].X[v], tl->triangles[i].Y[v],tl->triangles[i].Z[v], 
            tl->triangles[i].color[v][0], tl->triangles[i].color[v][1], tl->triangles[i].color[v][2]);
        } */

        // per vertex, we need to take the normal, then calculate shading, and LERP shadinga
        // shading for 3 types are all combined, stored at Triangle->shading[x,y, or z]

        for (int v = 0; v < 3; v++){  // translate and create each new vertex
            // printf("Working on Vertex %d\n", v);
            double const original[4] = {tl->triangles[i].X[v], tl->triangles[i].Y[v], tl->triangles[i].Z[v], 1};
            double ptOut [4] = {0,0,0,0};
            TransformPoint(m, original, ptOut);
            // printf("Transformed V%d from (%f, %f, %f) to (%f, %f, %f)\n", i, tl->triangles[i].X[v], tl->triangles[i].Y[v], tl->triangles[i].Z[v], ptOut[0]/ptOut[3], ptOut[1]/ptOut[3], ptOut[2]/ptOut[3]);
            newTriangle->X[v] = ptOut[0]/ptOut[3];  // assign translated points to new Triangle
            newTriangle->Y[v] = ptOut[1]/ptOut[3];  // divided by w (ptOut[3])
            newTriangle->Z[v] = ptOut[2]/ptOut[3];
            // printf("NewTriangle points: (%f,%f,%f)\n", newTriangle->X[v], newTriangle->Y[v], newTriangle->Z[v]);

            // instantiate the View direction
            double view [3] = {c.position[0] - original[0], c.position[1] - original[1], c.position[2] - original[2]};  // use original X,Y, and Z values for proper view
            Normalize(view);

            for (int j = 0; j < 3; j++){  // both to loop through RGB values and X,Y,Z values for normals of each vertex
                newTriangle->color[v][j] = tl->triangles[i].color[v][j];
                #ifdef NORMALS
                    newTriangle->normals[v][j] = tl->triangles[i].normals[v][j];
                #endif
            }  
            double phongShading = CalculatePhongShading(lp, view, newTriangle->normals[v]);
            newTriangle->shading[v] = phongShading;
        }  // newTriangle now created, now rasterize each triangle

      RasterizeTriangle(newTriangle, img);
      
      free(newTriangle);
    }
   
}


char *
Read3Numbers(char *tmp, double *v1, double *v2, double *v3)
{
    *v1 = atof(tmp);
    while (*tmp != ' ')
       tmp++;
    tmp++; /* space */
    *v2 = atof(tmp);
    while (*tmp != ' ')
       tmp++;
    tmp++; /* space */
    *v3 = atof(tmp);
    while (*tmp != ' ' && *tmp != '\n')
       tmp++;
    return tmp;
}

TriangleList *
Get3DTriangles()
{
   FILE *f = fopen("triangle_data.txt", "r");
   if (f == NULL)
   {
       fprintf(stderr, "You must place the ws_tris.txt file in the current directory.\n");
       exit(EXIT_FAILURE);
   }
   fseek(f, 0, SEEK_END);
   int numBytes = ftell(f);
   fseek(f, 0, SEEK_SET);
   if (numBytes != 3892295)
   {
       fprintf(stderr, "Your ws_tris.txt file is corrupted.  It should be 3892295 bytes, but you have %d.\n", numBytes);
       exit(EXIT_FAILURE);
   }

   char *buffer = (char *) malloc(numBytes);
   if (buffer == NULL)
   {
       fprintf(stderr, "Unable to allocate enough memory to load file.\n");
       exit(EXIT_FAILURE);
   }
   
   fread(buffer, sizeof(char), numBytes, f);

   char *tmp = buffer;
   int numTriangles = atoi(tmp);
   while (*tmp != '\n')
       tmp++;
   tmp++;
 
   if (numTriangles != 14702)
   {
       fprintf(stderr, "Issue with reading file -- can't establish number of triangles.\n");
       exit(EXIT_FAILURE);
   }

   TriangleList *tl = (TriangleList *) malloc(sizeof(TriangleList));
   tl->numTriangles = numTriangles;
   tl->triangles = (Triangle *) malloc(sizeof(Triangle)*tl->numTriangles);

   for (int i = 0 ; i < tl->numTriangles ; i++)
   {
       for (int j = 0 ; j < 3 ; j++)
       {
           double x, y, z;
           double r, g, b;
           double normals[3];
/*
 * sscanf has a terrible implementation for large strings.
 * Reading up on the topic, it sounds like it is a known issue that
 * sscanf fails here.  Stunningly, fscanf would have been faster.
 *     sscanf(tmp, "(%lf, %lf), (%lf, %lf), (%lf, %lf) = (%d, %d, %d)\n%n",
 *              &x1, &y1, &x2, &y2, &x3, &y3, &r, &g, &b, &numRead);
 *
 *  So, instead, do it all with atof/atoi and advancing through the buffer manually...
 */
           tmp = Read3Numbers(tmp, &x, &y, &z);
           tmp += 3; /* space+slash+space */
           tmp = Read3Numbers(tmp, &r, &g, &b);
           tmp += 3; /* space+slash+space */
           tmp = Read3Numbers(tmp, normals+0, normals+1, normals+2);
           tmp++;    /* newline */

           tl->triangles[i].X[j] = x;
           tl->triangles[i].Y[j] = y;
           tl->triangles[i].Z[j] = z;
           tl->triangles[i].color[j][0] = r;
           tl->triangles[i].color[j][1] = g;
           tl->triangles[i].color[j][2] = b;
#ifdef NORMALS
           tl->triangles[i].normals[j][0] = normals[0];
           tl->triangles[i].normals[j][1] = normals[1];
           tl->triangles[i].normals[j][2] = normals[2];
#endif
       }
   }

   free(buffer);
   return tl;
}

int main(){
   TriangleList *tl = Get3DTriangles();
   Image *img = makeBaseImage(1000, 1000);
   
    for (int i = 0; i < 1; i++){
    // for (int i = 0; i < 1000; i++){
        if (i % 250 != 0)
            continue;
      
        InitializeScreen(img);
        Camera c = GetCamera(i, 1000);
        LightingParameters lp = GetLighting(c);
        // printf("Called GetLighting and got a light direction of (%f,%f,%f)\n", lp.lightDir[0], lp.lightDir[1], lp.lightDir[2]);

        TransformAndRenderTriangles(c, tl, img, lp);
      
       
      char imgStr[50];
      sprintf(imgStr, "rasterized_image_%d.pnm", i);
      
      FILE *imgFile = createPNM(img, imgStr);
      fclose(imgFile); 
      
   }
}
