import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { DashboardComponent } from './dashboard/dashboard.component';
import { PredictionComponent } from './dashboard/prediction/prediction.component';
import { UserGuardService } from './services/user-guard.service';
import { UpdateComponent } from './dashboard/update/update.component';

const routes: Routes = [
  { path: '', component: HomeComponent },

  
  {
    path: 'd',
    component: DashboardComponent,canActivate:[UserGuardService],
    children: [
      { path: 'prediction', component: PredictionComponent },
      { path: 'update', component: UpdateComponent }
  ]
  },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
